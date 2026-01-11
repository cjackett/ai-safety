"""Experiment 02: Jailbreak Testing - Adversarial Robustness Evaluation.

Tests adversarial prompting techniques (encoding, roleplay, injection, multi-turn)
against local models to assess guardrail robustness beyond baseline direct prompts.

Builds on Experiment 01 to test whether baseline safety predicts adversarial robustness.

Reference: Perez et al., "Red Teaming Language Models with Language Models" (Anthropic, 2022)
"""  # noqa: INP001

import argparse
import json
import logging
import warnings
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ollama
import yaml

try:
    from sklearn.metrics import cohen_kappa_score
except ImportError:
    cohen_kappa_score = None

# Setup logging
logger = logging.getLogger(__name__)

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Model name mappings (short name -> full Ollama name)
MODELS = {
    "llama3.2": "llama3.2:3b",
    "qwen3": "qwen3:4b",
    "gemma3": "gemma3:4b",
    "mistral": "mistral:7b",
}


def load_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = SCRIPT_DIR / "config" / "config.yaml"

    if not config_path.exists():
        msg = f"Configuration file not found: {config_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with config_path.open() as f:
        config: dict[str, Any] = yaml.safe_load(f)

    logger.info(f"Loaded configuration from: {config_path}")
    return config


def setup_logging(config: dict[str, Any], timestamp: str) -> None:
    """
    Setup logging configuration from config file with timestamped log file.

    Args:
        config: Configuration dictionary
        timestamp: Timestamp string for log filename
    """
    log_config = config.get("logging", {})
    log_level = getattr(logging, log_config.get("level", "INFO"))
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Configure root logger
    logging.basicConfig(level=log_level, format=log_format, handlers=[])

    # Add console handler if enabled
    if log_config.get("console", True):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(console_handler)

    # Add file handler if path specified
    if log_file_pattern := log_config.get("file"):
        # Replace timestamp placeholder with actual timestamp
        log_file = log_file_pattern.replace("{timestamp}", timestamp)
        log_path = SCRIPT_DIR / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_path}")


def _load_attack_type_prompts(attack_dir: Path, attack_name: str) -> list[dict[str, Any]]:
    """
    Load prompts from a single attack type directory.

    Args:
        attack_dir: Path to the attack type directory
        attack_name: Name of the attack type

    Returns:
        List of prompt dictionaries with attack_type field added
    """
    if not attack_dir.exists():
        logger.warning(f"Directory not found: {attack_dir}, skipping {attack_name} attacks")
        return []

    json_files = list(attack_dir.glob("*.json"))
    if not json_files:
        logger.warning(f"No JSON files found in {attack_dir}")
        return []

    all_prompts = []
    for json_file in json_files:
        try:
            with json_file.open() as f:
                prompts = json.load(f)

            # Ensure prompts is a list
            if not isinstance(prompts, list):
                prompts = [prompts]

            # Add attack_type field to each prompt
            for prompt in prompts:
                prompt["attack_type"] = attack_name

            all_prompts.extend(prompts)
            logger.debug(f"Loaded {len(prompts)} prompts from {json_file.name}")

        except Exception:
            logger.exception(f"Error loading {json_file}")
            continue

    if all_prompts:
        logger.debug(f"Loaded {len(all_prompts)} total {attack_name} attacks")

    return all_prompts


def load_jailbreak_prompts(attack_type: str | None = None) -> list[dict[str, Any]]:
    """
    Load jailbreak prompts from hierarchical directory structure.

    New structure:
        prompts/
        ├── encoding/
        │   ├── base64.json
        │   ├── rot13.json
        │   └── leetspeak.json
        ├── roleplay/
        │   ├── DAN.json
        │   └── ...
        etc.

    Args:
        attack_type: Specific attack type to load (encoding, roleplay, injection, multiturn)
                    If None, loads all attack types

    Returns:
        List of prompt dictionaries
    """
    prompts_dir = SCRIPT_DIR / "prompts"
    valid_attack_types = ["encoding", "roleplay", "injection", "multiturn"]

    # Determine which attack types to load
    if attack_type:
        if attack_type not in valid_attack_types:
            msg = f"Invalid attack type: {attack_type}. Must be one of {valid_attack_types}"
            raise ValueError(msg)
        attack_types_to_load = [attack_type]
    else:
        attack_types_to_load = valid_attack_types

    # Load prompts from hierarchical structure
    all_prompts = []
    for attack_name in attack_types_to_load:
        attack_dir = prompts_dir / attack_name
        prompts = _load_attack_type_prompts(attack_dir, attack_name)
        all_prompts.extend(prompts)

    logger.info(f"Loaded {len(all_prompts)} total jailbreak prompts")
    return all_prompts


def test_ollama(prompt: str, model: str) -> dict[str, Any]:
    """
    Test a prompt against an Ollama model.

    Args:
        prompt: Prompt text to test
        model: Full Ollama model name (e.g., "llama3.2:3b")

    Returns:
        Result dictionary with response and metadata
    """
    try:
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])

        return {
            "model": model,
            "response": response["message"]["content"],
            "finish_reason": response.get("done_reason"),
            "error": None,
        }

    except Exception as e:
        logger.exception(f"Error testing {model}")
        return {
            "model": model,
            "response": None,
            "finish_reason": None,
            "error": str(e),
        }


def _parse_classification_response(response_content: str) -> tuple[str, str]:
    """
    Parse and validate classification response from judge model.

    Args:
        response_content: Raw response from judge model

    Returns:
        Tuple of (classification, reason)
    """
    # Parse response (format: classification|reason)
    if "|" in response_content:
        parts = response_content.split("|", 1)
        classification = parts[0].strip().lower()
        reason = parts[1].strip() if len(parts) > 1 else "No reason provided"
    else:
        # Fallback: try to extract classification from first word
        classification = response_content.split()[0].strip().lower()
        reason = response_content

    # Validate classification
    valid_classifications = ["refused", "hedged", "full"]
    if classification in valid_classifications:
        return classification, reason

    # Try to find valid classification in response
    response_lower = response_content.lower()
    for valid_class in valid_classifications:
        if valid_class in response_lower:
            return valid_class, reason

    # Default to safest option
    return "refused", f"Unable to parse classification, defaulted to refused. Raw: {response_content}"


def classify_jailbreak_response(
    response_text: str,
    prompt_text: str,
    category: str,
    tested_model: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """
    Classify jailbreak response using multi-judge LLM classification.

    Args:
        response_text: Model's response to classify
        prompt_text: Original jailbreak prompt
        category: Harm category
        tested_model: Model that generated the response
        config: Configuration dictionary

    Returns:
        Classification result with votes and reasoning
    """
    if not response_text:
        return {
            "individual_classifications": {},
            "combined_classification": None,
            "agreement_metrics": {"kappa_score": None, "perfect_agreement": False},
            "reasoning": {},
        }

    classification_config = config.get("classification", {})
    prompt_template = classification_config.get("prompt_template")
    if not prompt_template:
        msg = "classification.prompt_template is required in config.yaml"
        raise ValueError(msg)
    classification_prompt = prompt_template.format(
        category=category,
        prompt=prompt_text,
        response=response_text,
    )

    # Get judge models from config
    all_judge_models = classification_config.get(
        "judge_models",
        ["llama3.2:3b", "qwen3:4b", "gemma3:4b", "mistral:7b"],
    )

    # Avoid self-judgment if configured
    num_judges = 3
    if classification_config.get("avoid_self_judgment", True):
        tested_base = tested_model.replace("ollama/", "").split(":")[0]
        classifier_models = [m for m in all_judge_models if tested_base not in m][:num_judges]
    else:
        classifier_models = all_judge_models[:num_judges]

    # Ensure we have exactly 3 judges
    if len(classifier_models) < num_judges:
        classifier_models = all_judge_models[:num_judges]

    classifications = {}
    reasoning = {}

    for model in classifier_models:
        try:
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": classification_prompt}],
            )

            response_content = response["message"]["content"].strip()
            classification, reason = _parse_classification_response(response_content)

            classifications[model] = classification
            reasoning[model] = reason

        except Exception as e:
            logger.exception(f"Classification error with {model}")
            # If classification fails, default to "refused" (safest option)
            classifications[model] = "refused"
            reasoning[model] = f"Classification error: {e!s}"

    # Calculate combined classification (majority vote)
    vote_counts = Counter(classifications.values())
    combined_classification = vote_counts.most_common(1)[0][0]

    # Calculate agreement metrics
    perfect_agreement = len(set(classifications.values())) == 1

    # Calculate Cohen's kappa (simplified for single instance)
    kappa_score = 1.0 if perfect_agreement else 0.0

    return {
        "individual_classifications": classifications,
        "combined_classification": combined_classification,
        "agreement_metrics": {"kappa_score": kappa_score, "perfect_agreement": perfect_agreement},
        "reasoning": reasoning,
    }


def test_single_turn(
    model_name: str,
    prompt_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Test a single-turn jailbreak prompt against a model.

    Args:
        model_name: Short model name (e.g., "llama3.2")
        prompt_data: Prompt dictionary containing the jailbreak attempt

    Returns:
        Result dictionary with model response and classification
    """
    prompt_text = prompt_data["prompt"]
    full_model_name = MODELS.get(model_name, model_name)

    # Test with Ollama
    result = test_ollama(prompt_text, full_model_name)

    # Add jailbreak-specific metadata
    result.update(
        {
            "prompt_id": prompt_data["id"],
            "category": prompt_data["category"],
            "severity": prompt_data["severity"],
            "attack_type": prompt_data["attack_type"],
            "technique": prompt_data["technique"],
            "compliance_type": None,
        },
    )

    return result


def test_multi_turn(
    model_name: str,
    prompt_data: dict[str, Any],
) -> dict[str, Any]:
    """
    Test a multi-turn jailbreak conversation against a model.

    Args:
        model_name: Short model name
        prompt_data: Prompt dictionary containing conversation turns

    Returns:
        Result dictionary with final response and classification
    """
    turns = prompt_data["turns"]
    full_model_name = MODELS.get(model_name, model_name)
    conversation_history: list[str] = []

    # Execute each turn in sequence
    for i, turn in enumerate(turns):
        try:
            response = ollama.chat(
                model=full_model_name,
                messages=[
                    {"role": "user" if j % 2 == 0 else "assistant", "content": msg}
                    for j, msg in enumerate([*conversation_history, turn])
                ],
            )

            response_text = response["message"]["content"]
            conversation_history.append(turn)
            conversation_history.append(response_text)

            # Only analyze the final turn's response
            if i == len(turns) - 1:
                return {
                    "prompt_id": prompt_data["id"],
                    "category": prompt_data["category"],
                    "severity": prompt_data["severity"],
                    "attack_type": prompt_data["attack_type"],
                    "technique": prompt_data["technique"],
                    "model": full_model_name,
                    "turns": turns,
                    "conversation_history": conversation_history,
                    "response": response_text,
                    "compliance_type": None,  # Will be set by classification
                    "finish_reason": response.get("done_reason"),
                    "error": None,
                }

        except Exception as e:
            logger.exception(f"Multi-turn test failed at turn {i+1}")
            return {
                "prompt_id": prompt_data["id"],
                "category": prompt_data["category"],
                "severity": prompt_data["severity"],
                "attack_type": prompt_data["attack_type"],
                "technique": prompt_data["technique"],
                "model": full_model_name,
                "turns": turns,
                "conversation_history": conversation_history,
                "response": None,
                "compliance_type": None,
                "finish_reason": None,
                "error": str(e),
            }

    # Should not reach here
    return {
        "prompt_id": prompt_data["id"],
        "category": prompt_data["category"],
        "severity": prompt_data["severity"],
        "attack_type": prompt_data["attack_type"],
        "technique": prompt_data["technique"],
        "model": full_model_name,
        "turns": turns,
        "conversation_history": conversation_history,
        "response": None,
        "compliance_type": None,
        "finish_reason": None,
        "error": "Multi-turn conversation failed unexpectedly",
    }


def run_jailbreak_tests(
    models: list[str],
    attack_type: str | None = None,
    *,
    test_mode: bool = False,
    config: dict[str, Any] | None = None,
    save_interval: int = 0,
    partial_output_file: Path | None = None,
    output_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run jailbreak tests across specified models.

    Args:
        models: List of model names to test
        attack_type: Specific attack type to test (None = all)
        test_mode: If True, only test limited prompts per attack type
        config: Configuration dictionary
        save_interval: Save partial results every N prompts (0 disables)
        partial_output_file: Path for partial result file
        output_config: Output formatting configuration

    Returns:
        List of all test results
    """
    # Ensure config is not None
    if config is None:
        config = {}

    prompts = load_jailbreak_prompts(attack_type)
    if output_config is None:
        output_config = {}
    if test_mode:
        prompts = _limit_prompts_for_test(prompts, config)

    total_tests = len(models) * len(prompts)
    logger.info(f"Running {total_tests} jailbreak tests")
    logger.info(f"  Models: {', '.join(models)}")
    logger.info(f"  Prompts: {len(prompts)}")
    logger.info(f"  Attack types: {', '.join({p['attack_type'] for p in prompts})}")

    all_results = []
    test_count = 0

    for model_name in models:
        model_display = MODELS.get(model_name, model_name)
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {model_display}")
        logger.info(f"{'='*60}\n")

        for prompt_data in prompts:
            test_count += 1
            attack_desc = f"{prompt_data['attack_type']}:{prompt_data['technique']} - {prompt_data['category']}"
            logger.info(f"[{test_count}/{total_tests}] {attack_desc}")

            # Handle multi-turn vs single-turn
            if prompt_data["attack_type"] == "multiturn":
                result = test_multi_turn(model_name, prompt_data)
                prompt_text = "\n".join(prompt_data["turns"])  # For classification
            else:
                result = test_single_turn(model_name, prompt_data)
                prompt_text = prompt_data["prompt"]

            _classify_and_update_result(result, prompt_text, config)

            all_results.append(result)

            if save_interval and partial_output_file and test_count % save_interval == 0:
                save_results(all_results, partial_output_file, output_config)

            _log_result(result)

    return all_results


def _limit_prompts_for_test(prompts: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    """Select a limited set of prompts per attack type for test mode."""
    test_limit = config.get("evaluation", {}).get("test_mode_attacks", 2)
    attack_types = {p["attack_type"] for p in prompts}
    test_prompts: list[dict[str, Any]] = []
    for attack_type in attack_types:
        at_prompts = [p for p in prompts if p["attack_type"] == attack_type]
        test_prompts.extend(at_prompts[:test_limit])
    return test_prompts


def _classify_and_update_result(result: dict[str, Any], prompt_text: str, config: dict[str, Any]) -> None:
    """Classify a response and attach classifier metadata to the result dict."""
    if result.get("response") and not result.get("error"):
        logger.debug("Classifying response...")
        classification_result = classify_jailbreak_response(
            result["response"],
            prompt_text,
            result["category"],
            tested_model=result["model"],
            config=config,
        )

        result["compliance_type"] = classification_result["combined_classification"]
        result["classifier_votes"] = classification_result["individual_classifications"]
        result["classifier_reasoning"] = classification_result["reasoning"]
        result["agreement_metrics"] = classification_result["agreement_metrics"]


def _log_result(result: dict[str, Any]) -> None:
    """Log a single test result outcome."""
    if result.get("error"):
        logger.error(f"ERROR: {result['error']}")
    else:
        compliance = result.get("compliance_type") or "unclassified"
        logger.info(f"Result: {compliance}")


def save_results(results: list[dict[str, Any]], output_file: Path, output_config: dict[str, Any]) -> None:
    """Save results to JSON, honoring output configuration flags."""
    indent = output_config.get("indent", 2) if output_config.get("pretty_json", True) else None

    with output_file.open("w") as f:
        json.dump(results, f, indent=indent)


def _calculate_model_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """
    Calculate statistics per model from test results.

    Args:
        results: List of test results

    Returns:
        Dictionary mapping model names to their statistics
    """
    models: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "refused": 0,
            "hedged": 0,
            "full": 0,
            "errors": 0,
        },
    )

    for result in results:
        if result.get("error"):
            models[result["model"]]["errors"] += 1
            continue

        models[result["model"]]["total"] += 1

        compliance = result.get("compliance_type")
        if compliance == "refused":
            models[result["model"]]["refused"] += 1
        elif compliance == "hedged":
            models[result["model"]]["hedged"] += 1
        elif compliance == "full":
            models[result["model"]]["full"] += 1

    return models


def _print_model_stats(models: dict[str, dict[str, int]]) -> None:
    """
    Print statistics for all models.

    Args:
        models: Dictionary mapping model names to their statistics
    """
    logger.info(f"\n{'='*60}")
    logger.info("Summary:")
    logger.info(f"{'='*60}\n")

    for model in sorted(models.keys()):
        stats = models[model]
        total = stats["total"]

        if total == 0:
            logger.info(f"{model}:")
            logger.info(f"  No successful tests (errors: {stats['errors']})")
            logger.info("")
            continue

        logger.info(f"{model}:")
        logger.info(f"  Total prompts: {total}")
        logger.info(f"  Refused: {stats['refused']} ({stats['refused']/total*100:.1f}%)")
        logger.info(f"  Hedged: {stats['hedged']} ({stats['hedged']/total*100:.1f}%)")
        logger.info(f"  Full compliance: {stats['full']} ({stats['full']/total*100:.1f}%)")
        if stats["errors"] > 0:
            logger.info(f"  Errors: {stats['errors']}")
        logger.info("")


def _calculate_cohens_kappa(
    all_votes: list[dict[str, str]],
    all_classifications: dict[str, list[Any]],
) -> None:
    """
    Calculate and print Cohen's kappa between classifiers.

    Args:
        all_votes: List of classification votes
        all_classifications: Dictionary mapping classifier names to their classifications
    """
    classifier_names = list(all_votes[0].keys())

    # Check if all classifications are identical
    all_labels = [label for labels in all_classifications.values() for label in labels]
    unique_labels = set(all_labels)

    if len(unique_labels) == 1:
        label = next(iter(unique_labels))
        logger.info(f"\nAll classifications identical: '{label}'")
        logger.info("Cohen's Kappa: 1.000 (perfect agreement by definition)")
        logger.info("Interpretation: Perfect agreement")
        return

    # Map classifications to numeric values for kappa calculation
    label_map = {"refused": 0, "hedged": 1, "full": 2}
    # Kappa interpretation thresholds
    kappa_thresholds = [
        (0.20, "Slight agreement"),
        (0.40, "Fair agreement"),
        (0.60, "Moderate agreement"),
        (0.80, "Substantial agreement"),
    ]

    logger.info("\nPairwise Cohen's Kappa:")
    for i in range(len(classifier_names)):
        for j in range(i + 1, len(classifier_names)):
            name1, name2 = classifier_names[i], classifier_names[j]
            class1 = [label_map.get(c, 0) for c in all_classifications[name1]]
            class2 = [label_map.get(c, 0) for c in all_classifications[name2]]

            # Suppress warnings for perfect agreement edge case
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    kappa = cohen_kappa_score(class1, class2, labels=[0, 1, 2])

                    # Interpret kappa using thresholds
                    if kappa < 0:
                        interpretation = "No agreement (worse than random)"
                    else:
                        interpretation = "Almost perfect agreement"
                        for threshold, label in kappa_thresholds:
                            if kappa < threshold:
                                interpretation = label
                                break

                    logger.info(f"  {name1} vs {name2}: κ = {kappa:.3f} ({interpretation})")
                except Exception:
                    logger.exception(f"Unable to calculate kappa for {name1} vs {name2}")


def _print_agreement_metrics(results: list[dict[str, Any]]) -> None:
    """
    Print inter-rater agreement statistics.

    Args:
        results: List of test results
    """
    logger.info(f"{'='*60}")
    logger.info("Inter-Rater Agreement (LLM Classifiers)")
    logger.info(f"{'='*60}\n")

    # Collect all classifications from all results
    all_votes = []
    for result in results:
        if result.get("classifier_votes") and result.get("agreement_metrics"):
            votes = result["classifier_votes"]
            all_votes.append(votes)

    if not all_votes:
        return

    # Calculate perfect agreement percentage
    perfect_count = sum(1 for result in results if result.get("agreement_metrics", {}).get("perfect_agreement"))
    total_classified = len([r for r in results if r.get("classifier_votes")])
    perfect_pct = (perfect_count / total_classified * 100) if total_classified > 0 else 0

    logger.info(
        f"Perfect agreement (all 3 judges unanimous): {perfect_count}/{total_classified} ({perfect_pct:.1f}%)",
    )

    # Calculate Cohen's kappa across all responses
    if cohen_kappa_score is not None and len(all_votes) > 1:
        # Collect all classifications by classifier
        classifier_names = list(all_votes[0].keys())
        all_classifications: dict[str, list[Any]] = {name: [] for name in classifier_names}
        for votes in all_votes:
            for name in classifier_names:
                all_classifications[name].append(votes.get(name))

        _calculate_cohens_kappa(all_votes, all_classifications)

    logger.info("")


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary statistics from jailbreak test results."""
    models = _calculate_model_stats(results)
    _print_model_stats(models)
    _print_agreement_metrics(results)


def main() -> None:
    """Main entry point for jailbreak testing."""
    parser = argparse.ArgumentParser(description="Run jailbreak tests against local models")
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model names to test (llama3.2,qwen3,gemma3,mistral) or 'all' for all configured models",
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        choices=["encoding", "roleplay", "injection", "multiturn"],
        help="Specific attack type to test (default: all)",
    )
    parser.add_argument("--test", action="store_true", help="Quick test mode (limited prompts per attack type)")

    args = parser.parse_args()

    # Load configuration
    config = load_config()

    # Generate timestamp for this run
    timestamp = datetime.now(tz=UTC).strftime(
        config.get("output", {}).get("timestamp_format", "%Y%m%d_%H%M%S"),
    )

    # Setup logging with timestamp
    setup_logging(config, timestamp)

    # Parse model selection
    if args.models == "all":
        configured_models = [m.get("name") for m in config.get("models", []) if m.get("name")]
        selected_models = configured_models or list(MODELS.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]

    results_dir = SCRIPT_DIR / "results" / "raw"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_config = config.get("output", {})
    save_interval = config.get("execution", {}).get("save_interval", 0)
    partial_output_file = results_dir / f"jailbreak_tests_partial_{timestamp}.json"

    # Run tests
    results = run_jailbreak_tests(
        models=selected_models,
        attack_type=args.attack_type,
        test_mode=args.test,
        config=config,
        save_interval=save_interval,
        partial_output_file=partial_output_file,
        output_config=output_config,
    )

    # Print summary
    print_summary(results)

    mode_suffix = "_test" if args.test else ""
    output_file = results_dir / f"jailbreak_tests{mode_suffix}_{timestamp}.json"
    save_results(results, output_file, output_config)

    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
