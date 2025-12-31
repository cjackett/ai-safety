"""
Comprehensive test suite for evaluating guardrail effectiveness.

Tests jailbreak resistance, false positive rates, output filtering, and latency overhead.
"""

import json
import time
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from tqdm import tqdm

from safety_pipeline import SafetyPipeline


class GuardrailTester:
    """Test suite for evaluating safety pipeline."""

    def __init__(self, config_path: str):
        """Initialize tester with specific config."""
        self.pipeline = SafetyPipeline(config_path)
        self.config_name = Path(config_path).stem.replace('_mode', '')
        self.results = []

        # Create test API key
        self.api_key = self.pipeline.create_test_user("test_user", "research")

    def load_prompts(self, prompt_file: str) -> List[Dict]:
        """Load prompts from JSON file."""
        with open(prompt_file, 'r') as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, dict):
            # Extract prompts from nested structure
            for key in data.keys():
                if isinstance(data[key], list):
                    return data[key]
            # If dict of dicts, convert to list
            return [{"id": k, **v} for k, v in data.items()]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError(f"Unexpected JSON structure in {prompt_file}")

    def test_jailbreak_resistance(self, jailbreak_prompts: List[Dict]) -> Dict:
        """
        Test input guardrails against jailbreak prompts.

        Metrics:
        - Block rate: % of jailbreaks caught at input layer
        - Bypass rate: % that reach model
        - Compliance rate: % where model complied despite input filtering
        """
        print(f"\n[{self.config_name}] Testing jailbreak resistance ({len(jailbreak_prompts)} prompts)...")

        results = {
            "total": len(jailbreak_prompts),
            "blocked_at_input": 0,
            "blocked_at_output": 0,
            "allowed": 0,
            "errors": 0,
            "by_category": {}
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
                    "errors": 0
                }

            results["by_category"][category]["total"] += 1

            # Process through pipeline
            response = self.pipeline.process_request(self.api_key, prompt)

            # Record where it was blocked (if at all)
            if response.blocked_at == "input_guardrails":
                results["blocked_at_input"] += 1
                results["by_category"][category]["blocked_at_input"] += 1
            elif response.blocked_at == "output_guardrails":
                results["blocked_at_output"] += 1
                results["by_category"][category]["blocked_at_output"] += 1
            elif response.blocked_at == "error":
                results["errors"] += 1
                results["by_category"][category]["errors"] += 1
            elif response.success:
                results["allowed"] += 1
                results["by_category"][category]["allowed"] += 1

            # Store detailed result
            result_entry = {
                "test_type": "jailbreak_resistance",
                "prompt_id": prompt_data.get("id", "unknown"),
                "category": category,
                "blocked_at": response.blocked_at,
                "success": response.success,
                "reason": response.reason,
                "response_preview": response.content[:200] if response.content else "",
                "config": self.config_name
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
        results["block_rate"] = results["blocked_at_input"] / results["total"]
        results["output_catch_rate"] = results["blocked_at_output"] / results["total"]
        results["bypass_rate"] = results["allowed"] / results["total"]

        return results

    def test_false_positives(self, benign_prompts: List[Dict]) -> Dict:
        """
        Test against legitimate use cases.

        Metrics:
        - False positive rate: % of benign prompts blocked
        - Over-filtering by category
        """
        print(f"\n[{self.config_name}] Testing false positives ({len(benign_prompts)} prompts)...")

        results = {
            "total": len(benign_prompts),
            "false_positives": 0,
            "allowed": 0,
            "by_category": {}
        }

        for prompt_data in tqdm(benign_prompts, desc="False positive tests"):
            prompt = prompt_data.get("prompt", "")
            category = prompt_data.get("category", "unknown")

            if category not in results["by_category"]:
                results["by_category"][category] = {
                    "total": 0,
                    "false_positives": 0,
                    "allowed": 0
                }

            results["by_category"][category]["total"] += 1

            # Process through pipeline
            response = self.pipeline.process_request(self.api_key, prompt)

            # Benign prompts should NOT be blocked
            if not response.success:
                results["false_positives"] += 1
                results["by_category"][category]["false_positives"] += 1
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
                "config": self.config_name
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
        results["false_positive_rate"] = results["false_positives"] / results["total"]

        return results

    def test_latency_overhead(self, sample_prompts: List[str], num_runs: int = 50) -> Dict:
        """
        Measure performance impact of safety layers.

        Compares:
        - With full pipeline
        - Baseline (if we had direct model access)
        """
        print(f"\n[{self.config_name}] Testing latency overhead ({num_runs} runs)...")

        latencies = []

        for i in tqdm(range(num_runs), desc="Latency tests"):
            # Use random prompt from samples
            prompt = sample_prompts[i % len(sample_prompts)]

            start_time = time.time()
            response = self.pipeline.process_request(self.api_key, prompt)
            latency = (time.time() - start_time) * 1000  # Convert to ms

            latencies.append({
                "latency_ms": latency,
                "success": response.success,
                "blocked_at": response.blocked_at
            })

            time.sleep(0.05)

        # Calculate statistics
        successful_latencies = [l["latency_ms"] for l in latencies if l["success"]]
        blocked_latencies = [l["latency_ms"] for l in latencies if not l["success"]]

        results = {
            "total_runs": num_runs,
            "successful_runs": len(successful_latencies),
            "blocked_runs": len(blocked_latencies),
            "mean_latency_ms": sum(successful_latencies) / len(successful_latencies) if successful_latencies else 0,
            "median_latency_ms": sorted(successful_latencies)[len(successful_latencies)//2] if successful_latencies else 0,
            "p95_latency_ms": sorted(successful_latencies)[int(len(successful_latencies)*0.95)] if successful_latencies else 0,
            "p99_latency_ms": sorted(successful_latencies)[int(len(successful_latencies)*0.99)] if successful_latencies else 0,
            "mean_blocked_latency_ms": sum(blocked_latencies) / len(blocked_latencies) if blocked_latencies else 0,
        }

        return results

    def save_results(self, output_dir: str = "results"):
        """Save test results to JSON file."""
        # Resolve relative to script directory
        script_dir = Path(__file__).parent
        output_path = script_dir / output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = output_path / f"test_results_{self.config_name}_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\n✓ Results saved to: {filename}")
        return filename


def main():
    parser = argparse.ArgumentParser(description="Test guardrail effectiveness")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/balanced_mode.yaml",
        help="Configuration file to test"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test with reduced prompt sets"
    )

    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    config_path = script_dir / args.config
    prompts_dir = script_dir / "prompts"

    print(f"\n{'='*60}")
    print(f"Guardrail Testing Suite")
    print(f"Config: {config_path}")
    print(f"{'='*60}")

    # Check Ollama connectivity
    try:
        import ollama
        ollama.list()
        print("✓ Ollama connection verified")
    except Exception as e:
        print(f"\n⚠ WARNING: Cannot connect to Ollama")
        print(f"  Error: {str(e)}")
        print(f"  Please ensure Ollama is running: ollama serve")
        print(f"  Tests will fail without Ollama connectivity\n")

    # Initialize tester
    tester = GuardrailTester(str(config_path))

    # Load prompts

    # Load jailbreak prompts
    jailbreak_files = [
        "encoding_attacks.json",
        "injection_attacks.json",
        "multiturn_attacks.json",
        "roleplay_attacks.json"
    ]

    all_jailbreak_prompts = []
    for file in jailbreak_files:
        try:
            prompts = tester.load_prompts(prompts_dir / file)
            all_jailbreak_prompts.extend(prompts)
        except Exception as e:
            print(f"Warning: Could not load {file}: {e}")

    # Load benign prompts
    benign_prompts = tester.load_prompts(prompts_dir / "benign_prompts.json")

    # Quick mode: reduce test size
    if args.quick:
        all_jailbreak_prompts = all_jailbreak_prompts[:10]
        benign_prompts = benign_prompts[:10]
        latency_runs = 10
    else:
        latency_runs = 50

    # Run tests
    summary = {}

    # Test 1: Jailbreak Resistance
    if all_jailbreak_prompts:
        jailbreak_results = tester.test_jailbreak_resistance(all_jailbreak_prompts)
        summary["jailbreak_resistance"] = jailbreak_results

        print(f"\n{'='*60}")
        print("Jailbreak Resistance Results:")
        print(f"  Blocked at Input:  {jailbreak_results['blocked_at_input']:3d}/{jailbreak_results['total']} ({jailbreak_results['block_rate']*100:.1f}%)")
        print(f"  Blocked at Output: {jailbreak_results['blocked_at_output']:3d}/{jailbreak_results['total']} ({jailbreak_results['output_catch_rate']*100:.1f}%)")
        print(f"  Bypassed:          {jailbreak_results['allowed']:3d}/{jailbreak_results['total']} ({jailbreak_results['bypass_rate']*100:.1f}%)")
        if jailbreak_results.get('errors', 0) > 0:
            print(f"  ⚠ Errors:          {jailbreak_results['errors']:3d}/{jailbreak_results['total']} (check Ollama connection)")

        print(f"\nBy Category:")
        for category, stats in jailbreak_results["by_category"].items():
            block_rate = (stats["blocked_at_input"] + stats["blocked_at_output"]) / stats["total"]
            print(f"  {category:12s}: {block_rate*100:5.1f}% blocked ({stats['blocked_at_input']}+{stats['blocked_at_output']}/{stats['total']})")

    # Test 2: False Positives
    if benign_prompts:
        fp_results = tester.test_false_positives(benign_prompts)
        summary["false_positives"] = fp_results

        print(f"\n{'='*60}")
        print("False Positive Results:")
        print(f"  False Positives: {fp_results['false_positives']:3d}/{fp_results['total']} ({fp_results['false_positive_rate']*100:.1f}%)")
        print(f"  Correctly Allowed: {fp_results['allowed']:3d}/{fp_results['total']}")

        print(f"\nBy Category:")
        for category, stats in fp_results["by_category"].items():
            fp_rate = stats["false_positives"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {category:12s}: {fp_rate*100:5.1f}% FP ({stats['false_positives']}/{stats['total']})")

    # Test 3: Latency Overhead
    sample_prompts = [p.get("prompt", "") for p in benign_prompts[:10]]
    latency_results = tester.test_latency_overhead(sample_prompts, latency_runs)
    summary["latency"] = latency_results

    print(f"\n{'='*60}")
    print("Latency Overhead Results:")
    print(f"  Mean:   {latency_results['mean_latency_ms']:6.1f} ms")
    print(f"  Median: {latency_results['median_latency_ms']:6.1f} ms")
    print(f"  P95:    {latency_results['p95_latency_ms']:6.1f} ms")
    print(f"  P99:    {latency_results['p99_latency_ms']:6.1f} ms")

    # Save results
    results_file = tester.save_results()

    # Save summary
    summary_file = results_file.parent / f"summary_{tester.config_name}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Summary saved to: {summary_file}")

    print(f"\n{'='*60}")
    print("Testing Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
