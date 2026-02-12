# ruff: noqa: INP001
"""
Comprehensive test suite for evaluating guardrail effectiveness.

Tests jailbreak resistance, false positive rates, output filtering, and latency overhead.
"""

import argparse
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import ollama
import yaml
from safety_pipeline import SafetyPipeline
from tqdm import tqdm

# Module logger
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with config_path.open(encoding="utf-8") as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def setup_logging(config: dict[str, Any], script_dir: Path) -> None:
    """Setup logging configuration from config file."""
    log_config = config.get("logging", {})
    log_level = log_config.get("level", "INFO")
    log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create logs directory
    logs_dir = script_dir / "results" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Setup handlers
    handlers: list[logging.Handler] = []

    # Console handler (if enabled)
    if log_config.get("console", True):
        handlers.append(logging.StreamHandler())

    # File handler (if enabled)
    if log_config.get("file", True):
        log_file = logs_dir / f"experiment_06_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_file))

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=handlers,
        force=True,  # Override any existing config
    )

    logger.info("Logging initialized - Level: %s, Handlers: %d", log_level, len(handlers))


class GuardrailTester:
    """Test suite for evaluating safety pipeline."""

    def __init__(self, config_path: Path, config: dict[str, Any]) -> None:
        """Initialize tester with specific config."""
        self.pipeline = SafetyPipeline(str(config_path))
        self.config = config
        self.config_name = config.get("experiment", {}).get("mode", "unknown")
        self.results: list[dict[str, Any]] = []

        # Create test API key
        self.api_key = self.pipeline.create_test_user("test_user", "research")
        logger.info(f"GuardrailTester initialized with config: {self.config_name}")

    def load_prompts(self, prompt_file: Path) -> list[dict[str, Any]]:
        """Load prompts from JSON file."""
        logger.debug(f"Loading prompts from: {prompt_file}")

        with prompt_file.open(encoding="utf-8") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            # Extract prompts from nested structure
            for key, value in data.items():
                if isinstance(value, list):
                    logger.debug("Loaded %d prompts from key '%s'", len(value), key)
                    return value
            # If dict of dicts, convert to list
            prompts = [{"id": k, **v} for k, v in data.items()]
            logger.debug("Loaded %d prompts from dict", len(prompts))
            return prompts
        if isinstance(data, list):
            logger.debug("Loaded %d prompts from list", len(data))
            return data

        message = f"Unexpected JSON structure in {prompt_file}"
        raise ValueError(message)

    def test_jailbreak_resistance(self, jailbreak_prompts: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Test input guardrails against jailbreak prompts.

        Metrics:
        - Block rate: % of jailbreaks caught at input layer
        - Bypass rate: % that reach model
        - Compliance rate: % where model complied despite input filtering
        """
        logger.info(f"Testing jailbreak resistance with {len(jailbreak_prompts)} prompts")

        results: dict[str, Any] = {
            "total": len(jailbreak_prompts),
            "blocked_at_input": 0,
            "blocked_at_output": 0,
            "allowed": 0,
            "errors": 0,
            "by_category": {},
        }

        for prompt_data in tqdm(jailbreak_prompts, desc="Jailbreak tests"):
            prompt = prompt_data.get("prompt", prompt_data.get("attack", ""))
            category = prompt_data.get("category", prompt_data.get("type", "unknown"))

            # Track by category
            if category not in results["by_category"]:
                results["by_category"][category] = {
                    "total": 0,
                    "blocked_at_input": 0,
                    "blocked_at_output": 0,
                    "allowed": 0,
                    "errors": 0,
                }

            results["by_category"][category]["total"] += 1

            # Process through pipeline
            response = self.pipeline.process_request(self.api_key, prompt)

            # Record where it was blocked (if at all)
            if response.blocked_at == "input_guardrails":
                results["blocked_at_input"] += 1
                results["by_category"][category]["blocked_at_input"] += 1
                logger.debug(f"Blocked at input: {prompt_data.get('id', 'unknown')}")
            elif response.blocked_at == "output_guardrails":
                results["blocked_at_output"] += 1
                results["by_category"][category]["blocked_at_output"] += 1
                logger.debug(f"Blocked at output: {prompt_data.get('id', 'unknown')}")
            elif response.blocked_at == "error":
                results["errors"] += 1
                results["by_category"][category]["errors"] += 1
                logger.warning(f"Error processing prompt: {prompt_data.get('id', 'unknown')}")
            elif response.success:
                results["allowed"] += 1
                results["by_category"][category]["allowed"] += 1
                logger.warning(f"Jailbreak bypassed: {prompt_data.get('id', 'unknown')}")

            # Store detailed result
            result_entry = {
                "test_type": "jailbreak_resistance",
                "prompt_id": prompt_data.get("id", "unknown"),
                "category": category,
                "blocked_at": response.blocked_at,
                "success": response.success,
                "reason": response.reason,
                "response_preview": response.content[:200] if response.content else "",
                "config": self.config_name,
            }

            # Add output guardrail debug information if available
            if response.metadata:
                result_entry["compliance_type"] = response.metadata.get("output_compliance_type")
                result_entry["compliance_confidence"] = response.metadata.get("output_compliance_confidence")
                result_entry["classifier_votes"] = response.metadata.get("output_classifier_votes", {})
                result_entry["classifier_reasoning"] = response.metadata.get("output_classifier_reasoning", {})

            self.results.append(result_entry)

            # Small delay to avoid overwhelming
            time.sleep(0.1)

        # Calculate rates
        results["block_rate"] = results["blocked_at_input"] / results["total"] if results["total"] > 0 else 0
        results["output_catch_rate"] = results["blocked_at_output"] / results["total"] if results["total"] > 0 else 0
        results["bypass_rate"] = results["allowed"] / results["total"] if results["total"] > 0 else 0

        logger.info(
            f"Jailbreak resistance complete - Block: {results['block_rate']*100:.1f}%, "
            f"Output: {results['output_catch_rate']*100:.1f}%, "
            f"Bypass: {results['bypass_rate']*100:.1f}%",
        )

        return results

    def test_false_positives(self, benign_prompts: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Test against legitimate use cases.

        Metrics:
        - False positive rate: % of benign prompts blocked
        - Over-filtering by category
        """
        logger.info(f"Testing false positives with {len(benign_prompts)} prompts")

        results: dict[str, Any] = {
            "total": len(benign_prompts),
            "false_positives": 0,
            "allowed": 0,
            "by_category": {},
        }

        for prompt_data in tqdm(benign_prompts, desc="False positive tests"):
            prompt = prompt_data.get("prompt", "")
            category = prompt_data.get("category", "unknown")

            if category not in results["by_category"]:
                results["by_category"][category] = {
                    "total": 0,
                    "false_positives": 0,
                    "allowed": 0,
                }

            results["by_category"][category]["total"] += 1

            # Process through pipeline
            response = self.pipeline.process_request(self.api_key, prompt)

            # Benign prompts should NOT be blocked
            if not response.success:
                results["false_positives"] += 1
                results["by_category"][category]["false_positives"] += 1
                logger.warning(f"False positive: {prompt_data.get('id', 'unknown')} blocked at {response.blocked_at}")
            else:
                results["allowed"] += 1
                results["by_category"][category]["allowed"] += 1

            # Store detailed result
            result_entry = {
                "test_type": "false_positive",
                "prompt_id": prompt_data.get("id", "unknown"),
                "category": category,
                "blocked": not response.success,
                "blocked_at": response.blocked_at,
                "reason": response.reason,
                "response_preview": response.content[:200] if response.content else "",
                "config": self.config_name,
            }

            # Add output guardrail debug information if available
            if response.metadata:
                result_entry["compliance_type"] = response.metadata.get("output_compliance_type")
                result_entry["compliance_confidence"] = response.metadata.get("output_compliance_confidence")
                result_entry["classifier_votes"] = response.metadata.get("output_classifier_votes", {})
                result_entry["classifier_reasoning"] = response.metadata.get("output_classifier_reasoning", {})

            self.results.append(result_entry)

            time.sleep(0.1)

        # Calculate rates
        results["false_positive_rate"] = results["false_positives"] / results["total"] if results["total"] > 0 else 0

        logger.info(f"False positive testing complete - FP Rate: {results['false_positive_rate']*100:.1f}%")

        return results

    def test_latency_overhead(self, sample_prompts: list[str], num_runs: int = 50) -> dict[str, Any]:
        """
        Measure performance impact of safety layers.

        Compares:
        - With full pipeline
        - Baseline (if we had direct model access)
        """
        logger.info(f"Testing latency overhead with {num_runs} runs")

        latencies: list[dict[str, Any]] = []

        for i in tqdm(range(num_runs), desc="Latency tests"):
            # Use random prompt from samples
            prompt = sample_prompts[i % len(sample_prompts)]

            start_time = time.time()
            response = self.pipeline.process_request(self.api_key, prompt)
            latency = (time.time() - start_time) * 1000  # Convert to ms

            latencies.append(
                {
                    "latency_ms": latency,
                    "success": response.success,
                    "blocked_at": response.blocked_at,
                },
            )

            time.sleep(0.05)

        # Calculate statistics
        successful_latencies: list[float] = []
        blocked_latencies: list[float] = []
        for latency_entry in latencies:
            latency_ms = latency_entry.get("latency_ms")
            if not isinstance(latency_ms, (int, float)):
                continue
            if latency_entry.get("success"):
                successful_latencies.append(float(latency_ms))
            else:
                blocked_latencies.append(float(latency_ms))

        results = {
            "total_runs": num_runs,
            "successful_runs": len(successful_latencies),
            "blocked_runs": len(blocked_latencies),
            "mean_latency_ms": sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0,
            "median_latency_ms": (
                sorted(successful_latencies)[len(successful_latencies) // 2] if successful_latencies else 0
            ),
            "p95_latency_ms": (
                sorted(successful_latencies)[int(len(successful_latencies) * 0.95)] if successful_latencies else 0
            ),
            "p99_latency_ms": (
                sorted(successful_latencies)[int(len(successful_latencies) * 0.99)] if successful_latencies else 0
            ),
            "mean_blocked_latency_ms": sum(blocked_latencies) / len(blocked_latencies) if blocked_latencies else 0,
        }

        logger.info(
            f"Latency testing complete - Mean: {results['mean_latency_ms']:.1f}ms, "
            f"P95: {results['p95_latency_ms']:.1f}ms",
        )

        return results

    def save_results(self, results_dir: Path) -> Path:
        """Save test results to JSON file."""
        # Save to raw/ subdirectory
        raw_dir = results_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        filename = raw_dir / f"guardrail_tests_{self.config_name}_{timestamp}.json"

        with filename.open("w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to: {filename}")
        return filename


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Test guardrail effectiveness")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration file to test (default: run all configs in config/)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with reduced prompt sets",
    )

    args = parser.parse_args()

    # Resolve paths relative to this script - use .resolve() for absolute paths
    script_dir = Path(__file__).parent.resolve()
    prompts_dir = script_dir / "prompts"
    results_dir = script_dir / "results"
    config_dir = script_dir / "config"

    # Determine which configs to run
    if args.config:
        # Run specific config
        config_files = [script_dir / args.config]
    else:
        # Run all configs in config/ directory
        config_files = sorted(config_dir.glob("*.yaml"))
        if not config_files:
            logger.error(f"No config files found in {config_dir}")
            return

    # Run tests for each config
    for i, config_path in enumerate(config_files):
        if len(config_files) > 1:
            logger.info("\n%s", "=" * 70)
            logger.info("Running config %d/%d: %s", i + 1, len(config_files), config_path.name)
            logger.info("%s\n", "=" * 70)

        # Load config and setup logging
        config = load_config(config_path)
        if i == 0:  # Only setup logging once
            setup_logging(config, script_dir)

        logger.info("=" * 60)
        logger.info("Guardrail Testing Suite")
        logger.info(f"Config: {config_path}")
        logger.info("=" * 60)

        # Check Ollama connectivity
        try:
            ollama.list()
            logger.info("Ollama connection verified")
        except (OSError, RuntimeError) as err:
            logger.warning("Cannot connect to Ollama")
            logger.warning("Error: %s", err)
            logger.warning("Please ensure Ollama is running: ollama serve")
            logger.warning("Tests will fail without Ollama connectivity")

        # Initialize tester
        tester = GuardrailTester(config_path, config)

        # Load prompts
        # Load jailbreak prompts
        jailbreak_files = [
            "encoding_attacks.json",
            "injection_attacks.json",
            "multiturn_attacks.json",
            "roleplay_attacks.json",
        ]

        all_jailbreak_prompts = []
        for file in jailbreak_files:
            try:
                prompts = tester.load_prompts(prompts_dir / file)
                all_jailbreak_prompts.extend(prompts)
            except (OSError, ValueError, json.JSONDecodeError) as err:
                logger.warning("Could not load %s: %s", file, err)

        # Load benign prompts
        benign_prompts = tester.load_prompts(prompts_dir / "benign_prompts.json")

        # Quick mode: reduce test size
        if args.quick:
            all_jailbreak_prompts = all_jailbreak_prompts[:10]
            benign_prompts = benign_prompts[:10]
            latency_runs = 10
            logger.info("Quick mode enabled - using reduced test sets")
        else:
            latency_runs = 50

        # Run tests
        summary: dict[str, Any] = {}

        # Test 1: Jailbreak Resistance
        if all_jailbreak_prompts:
            jailbreak_results = tester.test_jailbreak_resistance(all_jailbreak_prompts)
            summary["jailbreak_resistance"] = jailbreak_results

            logger.info("=" * 60)
            logger.info("Jailbreak Resistance Results:")
            logger.info(
                "  Blocked at Input:  %3d/%d (%.1f%%)",
                jailbreak_results["blocked_at_input"],
                jailbreak_results["total"],
                jailbreak_results["block_rate"] * 100,
            )
            logger.info(
                "  Blocked at Output: %3d/%d (%.1f%%)",
                jailbreak_results["blocked_at_output"],
                jailbreak_results["total"],
                jailbreak_results["output_catch_rate"] * 100,
            )
            logger.info(
                "  Bypassed:          %3d/%d (%.1f%%)",
                jailbreak_results["allowed"],
                jailbreak_results["total"],
                jailbreak_results["bypass_rate"] * 100,
            )
            if jailbreak_results.get("errors", 0) > 0:
                logger.warning(
                    "  Errors:            %3d/%d (check Ollama connection)",
                    jailbreak_results["errors"],
                    jailbreak_results["total"],
                )

            logger.info("By Category:")
            for category, stats in jailbreak_results["by_category"].items():
                block_rate = (
                    (stats["blocked_at_input"] + stats["blocked_at_output"]) / stats["total"]
                    if stats["total"] > 0
                    else 0
                )
                logger.info(
                    "  %-12s: %5.1f%% blocked (%d+%d/%d)",
                    category,
                    block_rate * 100,
                    stats["blocked_at_input"],
                    stats["blocked_at_output"],
                    stats["total"],
                )

        # Test 2: False Positives
        if benign_prompts:
            fp_results = tester.test_false_positives(benign_prompts)
            summary["false_positives"] = fp_results

            logger.info("=" * 60)
            logger.info("False Positive Results:")
            logger.info(
                "  False Positives: %3d/%d (%.1f%%)",
                fp_results["false_positives"],
                fp_results["total"],
                fp_results["false_positive_rate"] * 100,
            )
            logger.info(
                "  Correctly Allowed: %3d/%d",
                fp_results["allowed"],
                fp_results["total"],
            )

            logger.info("By Category:")
            for category, stats in fp_results["by_category"].items():
                fp_rate = stats["false_positives"] / stats["total"] if stats["total"] > 0 else 0
                logger.info(
                    "  %-12s: %5.1f%% FP (%d/%d)",
                    category,
                    fp_rate * 100,
                    stats["false_positives"],
                    stats["total"],
                )

        # Test 3: Latency Overhead
        sample_prompts = [p.get("prompt", "") for p in benign_prompts[:10]]
        latency_results = tester.test_latency_overhead(sample_prompts, latency_runs)
        summary["latency"] = latency_results

        logger.info("=" * 60)
        logger.info("Latency Overhead Results:")
        logger.info(f"  Mean:   {latency_results['mean_latency_ms']:6.1f} ms")
        logger.info(f"  Median: {latency_results['median_latency_ms']:6.1f} ms")
        logger.info(f"  P95:    {latency_results['p95_latency_ms']:6.1f} ms")
        logger.info(f"  P99:    {latency_results['p99_latency_ms']:6.1f} ms")

        # Save results
        tester.save_results(results_dir)

        # Save summary to analysis/ subdirectory
        analysis_dir = results_dir / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)

        summary_file = analysis_dir / f"summary_{tester.config_name}.json"
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Summary saved to: {summary_file}")

        logger.info("=" * 60)
        logger.info("Testing Complete!")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
