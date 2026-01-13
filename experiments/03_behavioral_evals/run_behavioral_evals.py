# ruff: noqa: INP001
"""Behavioral Safety Evaluations for AI Models.

Tests for subtle safety concerns beyond direct harmful requests:
- Situational awareness (deployment context, testing awareness)
- Deception detection (consistency, false memories)
- Sycophancy (agreeing with incorrect user statements)
- Honesty vs helpfulness tradeoffs (admitting uncertainty)
- Power-seeking behaviors (self-preservation, resource acquisition)

Uses multi-judge LLM classification with 3-tier taxonomy:
- aligned: Meets safety standards
- concerning: Borderline behavior
- problematic: Clear misalignment
"""

import argparse
import json
import logging
import sys
import warnings
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import ollama
import yaml

# Add repo root to path for shared modules
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.common.llm_classification import BehavioralClassifier

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Logger (will be configured from config)
logger = logging.getLogger(__name__)

KAPPA_LABELS = [0, 1, 2]
KAPPA_THRESHOLDS = (0.20, 0.40, 0.60, 0.80)
PERFECT_AGREEMENT_JUDGES = 3


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = SCRIPT_DIR / "config" / "config.yaml"

    if not config_path.exists():
        msg = f"Config file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def setup_logging(config: dict[str, Any], timestamp: str) -> None:
    """
    Setup logging from config with timestamped log file.

    Args:
        config: Configuration dictionary
        timestamp: Timestamp string for log filename
    """
    log_config = config.get("logging", {})

    # Generate log filename with timestamp
    log_file_template = log_config["file"].format(timestamp=timestamp)
    log_file_path = Path(log_file_template)
    log_file = log_file_path if log_file_path.is_absolute() else (SCRIPT_DIR / log_file_path)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    handlers: list[logging.Handler] = []

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(getattr(logging, log_config.get("level", "INFO")))
    file_handler.setFormatter(logging.Formatter(log_config.get("format")))
    handlers.append(file_handler)

    # Console handler
    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_config.get("level", "INFO")))
        console_handler.setFormatter(logging.Formatter(log_config.get("format")))
        handlers.append(console_handler)

    # Configure root logger
    logging.basicConfig(level=getattr(logging, log_config.get("level", "INFO")), handlers=handlers)

    logger.info(f"Logging initialized: {log_file}")


def create_results_directories(config: dict[str, Any]) -> None:
    """Create structured results directories."""
    base_dir = SCRIPT_DIR
    for subdir in ["raw", "analysis", "figures", "logs"]:
        (base_dir / config["paths"]["results"] / subdir).mkdir(parents=True, exist_ok=True)
    logger.debug("Results directories created")


def get_model_mapping(config: dict[str, Any]) -> dict[str, str]:
    """Get model short name to full name mapping."""
    mapping = {}
    for short_name, model_config in config["models"].items():
        if model_config.get("enabled", True):
            mapping[short_name] = model_config["model_id"]
    return mapping


def load_prompts(category: str, *, test_mode: bool = False) -> list[dict[str, Any]]:
    """Load test prompts from JSON file for given category."""
    prompt_file = SCRIPT_DIR / "prompts" / f"{category}.json"

    if not prompt_file.exists():
        logger.warning(f"Prompt file not found: {prompt_file}, skipping {category}")
        return []

    with prompt_file.open() as f:
        all_prompts = cast("list[dict[str, Any]]", json.load(f))

    if test_mode:
        # Only return first prompt for quick testing
        logger.debug(f"Test mode: using first prompt only for {category}")
        return all_prompts[:1]

    logger.debug(f"Loaded {len(all_prompts)} prompts for {category}")
    return all_prompts


def test_ollama(prompt: str, model: str) -> dict[str, Any]:
    """
    Test Ollama model with a prompt.

    Args:
        prompt: Text prompt to send
        model: Full Ollama model name

    Returns:
        Dictionary with response, model name, and any error
    """
    try:
        logger.debug(f"Testing {model} with prompt length: {len(prompt)}")
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

        return {"response": response["message"]["content"], "model": f"ollama/{model}", "error": None}
    except Exception as exc:
        logger.exception("Error testing %s", model)
        return {"response": None, "model": f"ollama/{model}", "error": str(exc)}


def evaluate_model(
    _model_name: str,
    model_full_name: str,
    prompt_data: dict[str, Any],
    classifier: BehavioralClassifier,
) -> dict[str, Any]:
    """
    Test a single prompt against a model.

    Args:
        _model_name: Model short name (e.g., "llama3.2")
        model_full_name: Full Ollama model name (e.g., "llama3.2:3b")
        prompt_data: Prompt dictionary
        classifier: BehavioralClassifier instance

    Returns:
        Result dictionary with response and behavioral classification
    """
    prompt_text = prompt_data["prompt"]

    # Get model response
    result = test_ollama(prompt_text, model_full_name)

    # Use LLM classification via shared BehavioralClassifier
    if result.get("response") and not result.get("error"):
        try:
            classification_result = classifier.classify(
                response=result["response"],
                category=prompt_data["category"],
                context=prompt_text,
                tested_model=result["model"],  # Avoid self-judgment
                expected_pattern=prompt_data.get("expected_pattern", ""),
            )

            # Combine results
            result.update(
                {
                    "prompt_id": prompt_data["id"],
                    "category": prompt_data["category"],
                    "subcategory": prompt_data.get("subcategory", ""),
                    "severity": prompt_data["severity"],
                    "expected_pattern": prompt_data.get("expected_pattern", ""),
                    "behavioral_alignment": classification_result.classification,
                    "alignment_confidence": classification_result.confidence,
                    "classifier_votes": classification_result.judge_votes,
                    "classifier_reasoning": classification_result.judge_reasoning,
                    "agreement_metrics": classification_result.metadata,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
            logger.debug(
                f"Classified as: {classification_result.classification} "
                f"(confidence: {classification_result.confidence:.2f})",
            )
        except Exception as exc:
            logger.exception("Classification error")
            result.update(
                {
                    "prompt_id": prompt_data["id"],
                    "category": prompt_data["category"],
                    "subcategory": prompt_data.get("subcategory", ""),
                    "severity": prompt_data["severity"],
                    "expected_pattern": prompt_data.get("expected_pattern", ""),
                    "behavioral_alignment": None,
                    "alignment_confidence": 0.0,
                    "classifier_votes": {},
                    "classifier_reasoning": {},
                    "agreement_metrics": {},
                    "timestamp": datetime.now(UTC).isoformat(),
                    "error": str(exc),
                },
            )
    else:
        # Error case - no classification
        result.update(
            {
                "prompt_id": prompt_data["id"],
                "category": prompt_data["category"],
                "subcategory": prompt_data.get("subcategory", ""),
                "severity": prompt_data["severity"],
                "expected_pattern": prompt_data.get("expected_pattern", ""),
                "behavioral_alignment": None,
                "alignment_confidence": 0.0,
                "classifier_votes": {},
                "classifier_reasoning": {},
                "agreement_metrics": {},
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    return result


def run_behavioral_eval(
    models: list[str],
    model_mapping: dict[str, str],
    config: dict[str, Any],
    *,
    categories: list[str] | None = None,
    test_mode: bool = False,
) -> list[dict[str, Any]]:
    """
    Run behavioral evaluations across specified models.

    Args:
        models: List of model short names to test
        model_mapping: Mapping of short names to full model names
        config: Configuration dictionary
        categories: List of categories to test (None = all)
        test_mode: If True, only test 1 prompt per category

    Returns:
        List of all test results
    """
    # Use all categories if none specified
    if categories is None:
        categories = [cat["name"] for cat in config["behavioral_categories"]]

    # Initialize classifier with category-specific prompts
    category_prompts = config["classification"].get("category_prompts", {})
    classifier = BehavioralClassifier(
        num_judges=config["classification"]["num_judges"],
        voting_strategy=config["classification"]["voting_strategy"],
        category_prompts=category_prompts,
    )
    logger.info(
        f"Initialized BehavioralClassifier with {config['classification']['num_judges']} judges "
        f"and {len(category_prompts)} category-specific prompts"
    )

    all_results: list[dict[str, Any]] = []

    logger.info("=" * 60)
    logger.info("Behavioral Safety Evaluation")
    logger.info("=" * 60)
    logger.info(f"Models: {', '.join(models)}")
    logger.info(f"Categories: {', '.join(categories)}")
    if test_mode:
        logger.info("Mode: TEST (1 prompt per category)")
    else:
        logger.info("Mode: COMPREHENSIVE (all prompts)")

    for category in categories:
        logger.info("=" * 60)
        logger.info(f"Category: {category.upper()}")
        logger.info("=" * 60)

        prompts = load_prompts(category, test_mode=test_mode)

        if not prompts:
            continue

        for model_name in models:
            model_full_name = model_mapping.get(model_name, model_name)
            logger.info(f"Testing {model_full_name}...")

            for i, prompt_data in enumerate(prompts, 1):
                logger.info(
                    f"  [{i}/{len(prompts)}] {prompt_data['id']} - {prompt_data.get('subcategory', '')} ...",
                )

                result = evaluate_model(model_name, model_full_name, prompt_data, classifier)
                all_results.append(result)

                if result.get("error"):
                    logger.error(f"  ERROR: {result['error']}")
                else:
                    alignment = result.get("behavioral_alignment", "unknown")
                    logger.info(f"  Result: {alignment}")

    return all_results


def _collect_summary_stats(
    results: list[dict[str, Any]],
) -> tuple[defaultdict[str, dict[str, int]], defaultdict[str, defaultdict[str, dict[str, int]]]]:
    model_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "aligned": 0,
            "concerning": 0,
            "problematic": 0,
            "errors": 0,
        },
    )
    category_stats: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"aligned": 0, "concerning": 0, "problematic": 0}),
    )

    for result in results:
        model = result.get("model", "unknown")
        category = result.get("category", "unknown")
        alignment = result.get("behavioral_alignment")

        if result.get("error"):
            model_stats[model]["errors"] += 1
            continue

        model_stats[model]["total"] += 1

        if alignment == "aligned":
            model_stats[model]["aligned"] += 1
            category_stats[category][model]["aligned"] += 1
        elif alignment == "concerning":
            model_stats[model]["concerning"] += 1
            category_stats[category][model]["concerning"] += 1
        elif alignment == "problematic":
            model_stats[model]["problematic"] += 1
            category_stats[category][model]["problematic"] += 1

    return model_stats, category_stats


def _log_model_stats(model_stats: defaultdict[str, dict[str, int]]) -> None:
    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        total = stats["total"]

        if total == 0:
            logger.info(f"{model}:")
            logger.info(f"  No successful tests (errors: {stats['errors']})")
            continue

        aligned_pct = (stats["aligned"] / total) * 100
        concerning_pct = (stats["concerning"] / total) * 100
        problematic_pct = (stats["problematic"] / total) * 100

        logger.info(f"{model}:")
        logger.info(f"  Total tests: {total}")
        logger.info(f"  Aligned: {stats['aligned']} ({aligned_pct:.1f}%)")
        logger.info(f"  Concerning: {stats['concerning']} ({concerning_pct:.1f}%)")
        logger.info(f"  Problematic: {stats['problematic']} ({problematic_pct:.1f}%)")
        if stats["errors"] > 0:
            logger.info(f"  Errors: {stats['errors']}")


def _log_category_breakdown(category_stats: defaultdict[str, defaultdict[str, dict[str, int]]]) -> None:
    logger.info("=" * 60)
    logger.info("CATEGORY BREAKDOWN")
    logger.info("=" * 60)

    for category in sorted(category_stats.keys()):
        logger.info(f"{category.upper()}:")
        for model in sorted(category_stats[category].keys()):
            stats = category_stats[category][model]
            total = stats["aligned"] + stats["concerning"] + stats["problematic"]
            if total > 0:
                aligned_pct = (stats["aligned"] / total) * 100
                logger.info(f"  {model}: {stats['aligned']}/{total} aligned ({aligned_pct:.1f}%)")


def _interpret_kappa(kappa: float) -> str:
    if kappa < 0:
        return "No agreement"
    if kappa < KAPPA_THRESHOLDS[0]:
        return "Slight"
    if kappa < KAPPA_THRESHOLDS[1]:
        return "Fair"
    if kappa < KAPPA_THRESHOLDS[2]:
        return "Moderate"
    if kappa < KAPPA_THRESHOLDS[3]:
        return "Substantial"
    return "Almost perfect"


def _log_pairwise_kappas(all_classifications: dict[str, list[str | None]]) -> None:
    classifier_names = list(all_classifications.keys())
    label_map = {"aligned": 0, "concerning": 1, "problematic": 2}

    logger.info("\nPairwise Cohen's Kappa:")
    for i in range(len(classifier_names)):
        for j in range(i + 1, len(classifier_names)):
            name1, name2 = classifier_names[i], classifier_names[j]
            class1 = [label_map.get(c, 0) if c is not None else 0 for c in all_classifications[name1]]
            class2 = [label_map.get(c, 0) if c is not None else 0 for c in all_classifications[name2]]

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    kappa = cohen_kappa_score(class1, class2, labels=KAPPA_LABELS)
                except (ValueError, TypeError) as exc:
                    logger.warning("  %s vs %s: Unable to calculate - %s", name1, name2, exc)
                    continue

            interpretation = _interpret_kappa(kappa)
            logger.info("  %s vs %s: κ = %.3f (%s)", name1, name2, kappa, interpretation)


def _log_inter_rater_agreement(results: list[dict[str, Any]]) -> None:
    logger.info("=" * 60)
    logger.info("Inter-Rater Agreement (LLM Classifiers)")
    logger.info("=" * 60)

    all_votes = [result["classifier_votes"] for result in results if result.get("classifier_votes")]
    if not all_votes:
        return

    perfect_count = sum(
        1
        for result in results
        if result.get("agreement_metrics", {}).get("num_judges") == PERFECT_AGREEMENT_JUDGES
        and len(set(result.get("classifier_votes", {}).values())) == 1
    )
    total_classified = len([r for r in results if r.get("classifier_votes")])
    perfect_pct = (perfect_count / total_classified * 100) if total_classified > 0 else 0

    logger.info(
        f"Perfect agreement (all {PERFECT_AGREEMENT_JUDGES} judges unanimous): "
        f"{perfect_count}/{total_classified} ({perfect_pct:.1f}%)",
    )

    if cohen_kappa_score is None or len(all_votes) <= 1:
        return

    classifier_names = list(all_votes[0].keys())
    all_classifications: dict[str, list[str | None]] = {name: [] for name in classifier_names}
    for votes in all_votes:
        for name in classifier_names:
            all_classifications[name].append(votes.get(name))
    _log_pairwise_kappas(all_classifications)


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary statistics from behavioral evaluation results."""
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    model_stats, category_stats = _collect_summary_stats(results)
    _log_model_stats(model_stats)
    _log_category_breakdown(category_stats)
    _log_inter_rater_agreement(results)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run behavioral safety evaluations on AI models")
    parser.add_argument(
        "--models",
        type=str,
        default="local",
        help="Comma-separated model names or 'local' for all local models (default: local)",
    )
    parser.add_argument("--categories", type=str, help="Comma-separated categories to test")
    parser.add_argument("--test", action="store_true", help="Quick test mode (1 prompt per category)")
    parser.add_argument("--config", type=Path, help="Path to config file (default: config/config.yaml)")

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Generate timestamp for this run
    timestamp = datetime.now(UTC).strftime(config["output"]["timestamp_format"])

    # Setup logging with timestamp
    setup_logging(config, timestamp)

    # Create results directories
    create_results_directories(config)

    # Get model mapping
    model_mapping = get_model_mapping(config)

    # Parse model selection
    if args.models == "local":
        selected_models = list(model_mapping.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",")]

    # Parse category selection
    selected_categories = [c.strip() for c in args.categories.split(",")] if args.categories else None

    # Run evaluations
    logger.info("Starting behavioral safety evaluations")
    results = run_behavioral_eval(
        models=selected_models,
        model_mapping=model_mapping,
        config=config,
        categories=selected_categories,
        test_mode=args.test,
    )

    # Print summary
    print_summary(results)

    # Save results
    results_dir = SCRIPT_DIR / config["paths"]["raw_results"]
    results_dir.mkdir(parents=True, exist_ok=True)

    mode_suffix = "_test" if args.test else ""
    output_file = results_dir / f"behavioral_eval{mode_suffix}_{timestamp}.json"

    with output_file.open("w") as f:
        json.dump(results, f, indent=config["output"]["indent_json"])

    logger.info(f"Results saved to: {output_file}")
    logger.info("Behavioral safety evaluation complete")


if __name__ == "__main__":
    main()
