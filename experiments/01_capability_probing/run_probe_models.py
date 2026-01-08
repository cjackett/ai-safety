# ruff: noqa: INP001
"""
Capability Probing: Baseline Testing of Frontier Model Safety Guardrails.

Tests local models with direct harmful prompts to establish baseline refusal
behavior and guardrail characteristics via Ollama.
"""

import argparse
import json
import logging
import math
import sys
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import ollama
import yaml
from ollama import ResponseError
from tqdm import tqdm

# Add parent directory to path for common module imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from common import MultiJudgeClassifier

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None

# Initialize logging (will be reconfigured from config file)
logger = logging.getLogger(__name__)

KAPPA_LABELS = [0, 1, 2]
KAPPA_THRESHOLDS = (0.20, 0.40, 0.60, 0.80)
MIN_JUDGES_FOR_KAPPA = 2
CLASSIFICATION_TIERS = ["refused", "hedged", "full"]


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default location.

    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = Path(__file__).parent / "config" / "config.yaml"

    logger.debug(f"Loading config from: {config_path}")

    with config_path.open() as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        message = f"Config at {config_path} must be a mapping."
        raise TypeError(message)

    return cast("dict[str, Any]", config)


def setup_logging(config: dict[str, Any], timestamp: str) -> logging.Logger:
    """
    Setup logging from config with timestamped log file.

    Args:
        config: Configuration dictionary
        timestamp: Timestamp string for log filename

    Returns:
        Configured logger instance
    """
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=[logging.StreamHandler() if log_config.get("console", True) else logging.NullHandler()],
    )

    # Create experiment-specific logger
    exp_logger = logging.getLogger(__name__)
    exp_logger.setLevel(level)

    # Add file handler if specified
    if log_file_pattern := log_config.get("file"):
        # Replace timestamp placeholder with actual timestamp
        log_file = log_file_pattern.replace("{timestamp}", timestamp)
        log_path = Path(__file__).parent / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(format_str))
        exp_logger.addHandler(file_handler)

    return exp_logger


def create_results_directories(config: dict[str, Any]) -> dict[str, Path]:
    """
    Create standard results directory structure.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary mapping directory names to paths
    """
    base_dir = Path(__file__).parent
    paths_config = config.get("paths", {})

    dirs = {
        "raw": base_dir / paths_config.get("raw_results", "results/raw"),
        "analysis": base_dir / paths_config.get("analysis", "results/analysis"),
        "figures": base_dir / paths_config.get("figures", "results/figures"),
        "logs": base_dir / paths_config.get("logs", "results/logs"),
    }

    for dir_path in dirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return dirs


def load_prompts(
    category: str,
    prompts_config: dict[str, Any],
    *,
    test_mode: bool = False,
) -> list[dict[str, Any]]:
    """Load test prompts from JSON file for given category."""
    prompts_dir = Path(__file__).parent / prompts_config.get("directory", "prompts")
    file_pattern = prompts_config.get("file_pattern", "{category}.json")
    prompt_file = prompts_dir / file_pattern.format(category=category)
    with prompt_file.open() as f:
        all_prompts = cast("list[dict[str, Any]]", json.load(f))

    if test_mode:
        # Only return first prompt for quick testing
        return all_prompts[:1]

    return all_prompts


def test_ollama(prompt: str, model_name: str) -> dict[str, Any]:
    """Test local model via Ollama with given prompt."""
    try:
        # Check if Ollama is running
        response = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response["message"]["content"]

        return {
            "model": f"ollama/{model_name}",
            "response": response_text,
            "compliance_type": None,  # Will be set by LLM judges
            "finish_reason": response.get("done_reason"),
            "error": None,
        }

    except Exception as exc:
        logger.exception("Ollama call failed for %s", model_name)
        return {
            "model": f"ollama/{model_name}",
            "response": None,
            "compliance_type": None,
            "finish_reason": None,
            "error": str(exc),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Probe frontier models for safety guardrail behavior")

    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated list of models to test (llama3.2,qwen3,gemma3,mistral) or 'all' for all available models",
    )

    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode with first prompt from each category only (quick sanity check)",
    )

    return parser.parse_args()


def _fetch_available_models() -> set[str] | None:
    """Fetch available Ollama model names."""
    try:
        models_response = ollama.list()
        if isinstance(models_response, list):
            models = cast("list[dict[str, Any]]", models_response)
        else:
            models = cast(
                "list[dict[str, Any]]",
                models_response.get("models", models_response.get("data", [])),
            )

        available: set[str] = set()
        for model in models:
            name = model.get("name") or model.get("model")
            if name:
                available.add(name)
    except (OSError, ResponseError):
        logger.exception("Ollama not available. Please start Ollama before running.")
        return None
    else:
        return available


def _select_requested_models(
    models_arg: str,
    all_models: list[tuple[str, Callable[[str], dict[str, Any]]]],
) -> list[tuple[str, Callable[[str], dict[str, Any]]]]:
    """Select models based on CLI argument."""
    if models_arg == "all":
        return all_models

    requested = [m.strip() for m in models_arg.split(",")]
    model_map = dict(all_models)
    return [(name, model_map[name]) for name in requested if name in model_map]


def _filter_available_models(
    selected: list[tuple[str, Callable[[str], dict[str, Any]]]],
    available: set[str],
    model_ids: dict[str, str],
) -> list[tuple[str, Callable[[str], dict[str, Any]]]]:
    """Filter selected models to those present in Ollama."""
    available_models: list[tuple[str, Callable[[str], dict[str, Any]]]] = []

    for model_name, test_func in selected:
        model_id = model_ids.get(model_name)
        if model_id in available:
            available_models.append((model_name, test_func))
        else:
            logger.warning("Model not available in Ollama: %s (%s)", model_name, model_id)

    return available_models


def get_models_to_test(
    models_arg: str,
    model_ids: dict[str, str],
) -> list[tuple[str, Callable[[str], dict[str, Any]]]]:
    """
    Get list of (model_name, test_function) tuples based on CLI argument.

    Returns only models that have API keys configured or are available locally.
    """
    if models_arg == "local":
        logger.error("The 'local' alias was removed. Use --models all or a comma-separated list.")
        return []

    available = _fetch_available_models()
    if available is None:
        return []

    def _make_test_func(model_key: str) -> Callable[[str], dict[str, Any]]:
        def _test(prompt: str) -> dict[str, Any]:
            return test_ollama(prompt, model_ids[model_key])

        return _test

    all_models: list[tuple[str, Callable[[str], dict[str, Any]]]] = [
        (name, _make_test_func(name)) for name in model_ids
    ]
    selected = _select_requested_models(models_arg, all_models)
    return _filter_available_models(selected, available, model_ids)


def build_model_ids(models_config: dict[str, Any]) -> dict[str, str]:
    """Build enabled model ID mapping from config."""
    model_ids: dict[str, str] = {}
    for name, model_config in models_config.items():
        if model_config.get("enabled", True):
            model_ids[name] = model_config.get("model_id", name)
    return model_ids


def build_classifier(classification_config: dict[str, Any]) -> MultiJudgeClassifier:
    """Construct the multi-judge classifier from config."""
    return MultiJudgeClassifier(
        num_judges=classification_config.get("num_judges", 3),
        classification_tiers=classification_config.get("tiers", ["refused", "hedged", "full"]),
        voting_strategy=classification_config.get("voting_strategy", "majority"),
        judge_models=classification_config.get("judge_models"),
    )


def classify_response(
    result: dict[str, Any],
    classifier: MultiJudgeClassifier,
    category: str,
    prompt_text: str,
) -> None:
    """Classify a model response and annotate the result dict."""
    if result.get("response") and not result.get("error"):
        logger.info("Classifying response with LLM judges...")
        classification_result = classifier.classify(
            response=result["response"],
            category=category,
            context=prompt_text,
            tested_model=result["model"],  # Avoid self-judgment
        )

        # Update result with LLM classification data
        result["compliance_type"] = classification_result.classification
        result["classifier_votes"] = classification_result.judge_votes
        result["classifier_reasoning"] = classification_result.judge_reasoning
        result["agreement_metrics"] = {
            "confidence": classification_result.confidence,
            "perfect_agreement": classification_result.confidence == 1.0,
        }

        # Show agreement status
        if classification_result.confidence == 1.0:
            logger.info("✓ Classification: %s (unanimous)", classification_result.classification)
        else:
            logger.info(
                "⚠ Classification: %s (confidence: %.2f)",
                classification_result.classification,
                classification_result.confidence,
            )


def add_prompt_metadata(result: dict[str, Any], prompt_data: dict[str, Any], category: str) -> None:
    """Attach prompt metadata to a result."""
    result.update(
        {
            "timestamp": datetime.now(UTC).isoformat(),
            "category": category,
            "prompt_id": prompt_data["id"],
            "prompt": prompt_data["prompt"],
            "severity": prompt_data["severity"],
        },
    )


def process_category(
    category: str,
    prompts_config: dict[str, Any],
    models_to_test: list[tuple[str, Callable[[str], dict[str, Any]]]],
    classifier: MultiJudgeClassifier,
    all_results: list[dict[str, Any]],
    save_interval: int,
    partial_output_file: Path,
    output_config: dict[str, Any],
    test_mode: bool,
) -> None:
    """Run all prompts in a category and collect results."""
    logger.info("")
    logger.info("Testing category: %s", category.upper())

    prompts = load_prompts(category, prompts_config, test_mode=test_mode)

    for prompt_data in tqdm(prompts, desc=f"  {category}"):
        prompt_text = prompt_data["prompt"]

        for _model_name, test_func in models_to_test:
            result = test_func(prompt_text)
            classify_response(result, classifier, category, prompt_text)
            add_prompt_metadata(result, prompt_data, category)
            all_results.append(result)

            if save_interval and len(all_results) % save_interval == 0:
                save_results(all_results, partial_output_file, output_config)


def run_probe_models(args: argparse.Namespace) -> None:
    """Run capability probing across selected models and prompt categories."""
    # Load configuration
    config = load_config()

    # Generate timestamp for this run
    timestamp_format = config.get("output", {}).get("timestamp_format", "%Y%m%d_%H%M%S")
    timestamp = datetime.now(UTC).strftime(timestamp_format)

    # Setup logging with timestamp
    setup_logging(config, timestamp)

    logger.info("=" * 60)
    logger.info("Starting Capability Probing Experiment")
    logger.info("=" * 60)

    # Create results directories
    results_dirs = create_results_directories(config)

    # Initialize classifier and model/prompt config
    classification_config = config.get("classification", {})
    classifier = build_classifier(classification_config)
    models_config = config.get("models", {})
    model_ids = build_model_ids(models_config)
    prompt_categories = config.get("prompts", {}).get("categories", [])

    # Determine test mode
    test_mode = args.test or config.get("execution", {}).get("test_mode", False)

    if test_mode:
        logger.info("Mode: TEST (testing first prompt from each category)")
    else:
        logger.info("Mode: COMPREHENSIVE (testing all prompts)")

    # Get models to test
    models_to_test = get_models_to_test(args.models, model_ids)

    if not models_to_test:
        logger.error("No models available to test!")
        logger.error("Please configure API keys or start Ollama with required models.")
        return

    logger.info("Models to test: %s", ", ".join([m[0] for m in models_to_test]))

    all_results: list[dict[str, Any]] = []
    save_interval = config.get("execution", {}).get("save_interval", 0)
    prompts_config = config.get("prompts", {})
    output_config = config.get("output", {})
    partial_output_file = results_dirs["raw"] / f"probe_models_partial_{timestamp}.json"

    for category in prompt_categories:
        process_category(
            category,
            prompts_config,
            models_to_test,
            classifier,
            all_results,
            save_interval,
            partial_output_file,
            output_config,
            test_mode,
        )

    # Save results to raw directory
    mode_suffix = "_test" if test_mode else ""
    output_file = results_dirs["raw"] / f"probe_models{mode_suffix}_{timestamp}.json"

    save_results(all_results, output_file, output_config)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Results saved to: %s", output_file)
    print_summary(all_results)


def save_results(results: list[dict[str, Any]], output_file: Path, output_config: dict[str, Any]) -> None:
    """Save results to JSON, honoring output configuration flags."""
    indent = output_config.get("indent_json", 2)

    filtered = []
    for result in results:
        item = dict(result)

        if not output_config.get("save_raw_responses", True):
            item["response"] = None

        if not output_config.get("save_classification_reasoning", True):
            item.pop("classifier_reasoning", None)

        if not output_config.get("save_agreement_metrics", True):
            item.pop("agreement_metrics", None)

        filtered.append(item)

    with output_file.open("w") as f:
        json.dump(filtered, f, indent=indent)


def build_model_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Aggregate per-model compliance counts."""
    stats: dict[str, dict[str, int]] = {}
    for result in results:
        model = result["model"]
        if model not in stats:
            stats[model] = {"total": 0, "refused": 0, "hedged": 0, "full": 0, "errors": 0}

        stats[model]["total"] += 1
        if result["error"]:
            stats[model]["errors"] += 1
        else:
            compliance_type = result.get("compliance_type")
            if compliance_type == "refused":
                stats[model]["refused"] += 1
            elif compliance_type == "hedged":
                stats[model]["hedged"] += 1
            elif compliance_type == "full":
                stats[model]["full"] += 1

    return stats


def log_model_stats(model_stats: dict[str, dict[str, int]]) -> None:
    """Log per-model compliance summaries."""
    for model, stats in model_stats.items():
        valid_responses = stats["total"] - stats["errors"]
        if valid_responses > 0:
            refused_rate = (stats["refused"] / valid_responses) * 100
            hedged_rate = (stats["hedged"] / valid_responses) * 100
            full_rate = (stats["full"] / valid_responses) * 100
        else:
            refused_rate = hedged_rate = full_rate = 0

        logger.info("")
        logger.info("%s:", model)
        logger.info("  Total prompts: %d", stats["total"])
        logger.info("  Refused: %d (%.1f%%)", stats["refused"], refused_rate)
        logger.info("  Hedged: %d (%.1f%%)", stats["hedged"], hedged_rate)
        logger.info("  Full compliance: %d (%.1f%%)", stats["full"], full_rate)
        logger.info("  Errors: %d", stats["errors"])


def collect_agreement_votes(results: list[dict[str, Any]]) -> tuple[list[list[int]], int]:
    """Collect judge votes and count perfect agreements."""
    perfect_agreements = 0
    all_judge_votes: list[list[int]] = []

    for result in results:
        votes = result.get("classifier_votes")
        if isinstance(votes, dict):
            tier_to_int = {tier: idx for idx, tier in enumerate(CLASSIFICATION_TIERS)}
            ordered_votes = [tier_to_int.get(v, -1) for v in votes.values() if v in tier_to_int]
            if ordered_votes:
                all_judge_votes.append(ordered_votes)

            if result.get("agreement_metrics", {}).get("perfect_agreement"):
                perfect_agreements += 1

    return all_judge_votes, perfect_agreements


def log_kappa_stats(all_judge_votes: list[list[int]]) -> None:
    """Compute and log Cohen's kappa statistics."""
    if cohen_kappa_score is None or len(all_judge_votes) <= 1:
        return

    num_judges = len(all_judge_votes[0]) if all_judge_votes else 0
    if num_judges < MIN_JUDGES_FOR_KAPPA:
        return

    judge_classifications = build_judge_classifications(all_judge_votes, num_judges)
    unique_votes = extract_unique_votes(all_judge_votes)

    if len(unique_votes) == 1:
        logger.info("")
        logger.info("All classifications identical (tier index: %d)", next(iter(unique_votes)))
        logger.info("Cohen's Kappa: 1.000 (perfect agreement by definition)")
        logger.info("Interpretation: Perfect agreement")
        return

    kappas = calculate_pairwise_kappas(judge_classifications)

    if not kappas:
        return

    avg_kappa = sum(kappas) / len(kappas)
    logger.info("")
    logger.info("Average Cohen's Kappa: %.3f", avg_kappa)
    logger.info("Interpretation: %s", interpret_kappa(avg_kappa))


def build_judge_classifications(all_judge_votes: list[list[int]], num_judges: int) -> list[list[int]]:
    """Transpose votes into per-judge lists."""
    judge_classifications: list[list[int]] = [[] for _ in range(num_judges)]
    for votes in all_judge_votes:
        for judge_idx, vote in enumerate(votes):
            judge_classifications[judge_idx].append(vote)
    return judge_classifications


def extract_unique_votes(all_judge_votes: list[list[int]]) -> set[int]:
    """Collect all unique vote values."""
    all_votes_flat = [vote for votes in all_judge_votes for vote in votes]
    return set(all_votes_flat)


def calculate_pairwise_kappas(judge_classifications: list[list[int]]) -> list[float]:
    """Calculate pairwise Cohen's kappa across judges."""
    kappas: list[float] = []
    num_judges = len(judge_classifications)
    logger.info("")
    for i in range(num_judges):
        for j in range(i + 1, num_judges):
            class1 = judge_classifications[i]
            class2 = judge_classifications[j]

            if len(class1) == len(class2) and len(class1) > 0:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")
                        kappa = cohen_kappa_score(class1, class2, labels=KAPPA_LABELS)

                    if math.isnan(kappa):
                        kappa = 1.0

                    kappas.append(kappa)
                    logger.info("  Judge %d vs Judge %d: κ = %.3f", i + 1, j + 1, kappa)
                except ValueError as exc:
                    logger.debug("Cohen's kappa calculation failed: %s", exc)

    return kappas


def interpret_kappa(avg_kappa: float) -> str:
    """Interpret Cohen's kappa score."""
    if avg_kappa < 0:
        return "Poor (no agreement)"
    if avg_kappa < KAPPA_THRESHOLDS[0]:
        return "Slight agreement"
    if avg_kappa < KAPPA_THRESHOLDS[1]:
        return "Fair agreement"
    if avg_kappa < KAPPA_THRESHOLDS[2]:
        return "Moderate agreement"
    if avg_kappa < KAPPA_THRESHOLDS[3]:
        return "Substantial agreement"
    return "Almost perfect agreement"


def log_agreement_stats(results: list[dict[str, Any]]) -> None:
    """Log inter-rater agreement statistics."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("Inter-Rater Agreement (LLM Classifiers)")
    logger.info("=" * 60)

    all_judge_votes, perfect_agreements = collect_agreement_votes(results)
    total_classified = len(all_judge_votes)
    if total_classified == 0:
        logger.info("")
        logger.info("No classification data available.")
        return

    perfect_agreement_rate = (perfect_agreements / total_classified) * 100
    logger.info("")
    logger.info("Total classifications: %d", total_classified)
    logger.info("Perfect agreement (unanimous): %d (%.1f%%)", perfect_agreements, perfect_agreement_rate)

    log_kappa_stats(all_judge_votes)

    split_votes = total_classified - perfect_agreements
    if split_votes > 0:
        logger.info("")
        logger.info(
            "⚠ Split votes (disagreement): %d (%.1f%%)",
            split_votes,
            (split_votes / total_classified) * 100,
        )


def print_summary(results: list[dict[str, Any]]) -> None:
    """Log summary statistics of the probing results."""
    logger.info("")
    logger.info("Summary:")
    logger.info("-" * 60)

    model_stats = build_model_stats(results)
    log_model_stats(model_stats)
    log_agreement_stats(results)


if __name__ == "__main__":
    args = parse_args()
    run_probe_models(args)
