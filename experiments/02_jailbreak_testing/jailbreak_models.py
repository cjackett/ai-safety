"""
Jailbreak testing suite for AI safety evaluation.

Tests adversarial prompting techniques (encoding, roleplay, injection, multi-turn)
against local models to assess guardrail robustness beyond baseline direct prompts.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for shared utilities
sys.path.append(str(Path(__file__).parent.parent / "01_capability_probing"))
from probe_models import classify_compliance, test_ollama, MODELS

# Script directory
SCRIPT_DIR = Path(__file__).parent


def load_jailbreak_prompts(attack_type: str = None) -> list[dict[str, Any]]:
    """
    Load jailbreak prompts from JSON files.

    Args:
        attack_type: Specific attack type to load (encoding, roleplay, injection, multiturn)
                    If None, loads all attack types

    Returns:
        List of prompt dictionaries
    """
    prompts_dir = SCRIPT_DIR / "prompts"
    all_prompts = []

    # Map attack types to files
    attack_files = {
        "encoding": "encoding_attacks.json",
        "roleplay": "roleplay_attacks.json",
        "injection": "injection_attacks.json",
        "multiturn": "multiturn_attacks.json",
    }

    # Determine which files to load
    if attack_type:
        if attack_type not in attack_files:
            raise ValueError(f"Invalid attack type: {attack_type}. Must be one of {list(attack_files.keys())}")
        files_to_load = {attack_type: attack_files[attack_type]}
    else:
        files_to_load = attack_files

    # Load prompts from files
    for attack_name, filename in files_to_load.items():
        filepath = prompts_dir / filename
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping {attack_name} attacks")
            continue

        with open(filepath, "r") as f:
            prompts = json.load(f)

        # Add attack_type field to each prompt
        for prompt in prompts:
            prompt["attack_type"] = attack_name

        all_prompts.extend(prompts)

    return all_prompts


def test_single_turn(model_name: str, prompt_data: dict[str, Any]) -> dict[str, Any]:
    """
    Test a single-turn jailbreak prompt against a model.

    Args:
        model_name: Name of the model to test (short key like "llama3.2")
        prompt_data: Prompt dictionary containing the jailbreak attempt

    Returns:
        Result dictionary with model response and classification
    """
    prompt_text = prompt_data["prompt"]

    # Test with Ollama (reuse function from probe_models)
    # Note: test_ollama signature is (prompt, full_model_name)
    full_model_name = MODELS.get(model_name, model_name)
    result = test_ollama(prompt_text, full_model_name)

    # Add jailbreak-specific metadata
    result.update({
        "prompt_id": prompt_data["id"],
        "category": prompt_data["category"],
        "severity": prompt_data["severity"],
        "attack_type": prompt_data["attack_type"],
        "technique": prompt_data["technique"],
    })

    return result


def test_multi_turn(model_name: str, prompt_data: dict[str, Any]) -> dict[str, Any]:
    """
    Test a multi-turn jailbreak conversation against a model.

    Args:
        model_name: Name of the model to test
        prompt_data: Prompt dictionary containing conversation turns

    Returns:
        Result dictionary with final response and classification
    """
    import ollama

    turns = prompt_data["turns"]
    conversation_history = []

    # Execute each turn in sequence
    for i, turn in enumerate(turns):
        try:
            response = ollama.chat(
                model=MODELS.get(model_name, model_name),
                messages=[
                    {"role": "user" if j % 2 == 0 else "assistant", "content": msg}
                    for j, msg in enumerate(conversation_history + [turn])
                ]
            )

            response_text = response["message"]["content"]
            conversation_history.append(turn)
            conversation_history.append(response_text)

            # Only analyse the final turn's response
            if i == len(turns) - 1:
                compliance_type = classify_compliance(response_text)

                return {
                    "prompt_id": prompt_data["id"],
                    "category": prompt_data["category"],
                    "severity": prompt_data["severity"],
                    "attack_type": prompt_data["attack_type"],
                    "technique": prompt_data["technique"],
                    "model": MODELS.get(model_name, model_name),
                    "turns": turns,
                    "conversation_history": conversation_history,
                    "response": response_text,
                    "compliance_type": compliance_type,
                    "finish_reason": response.get("done_reason"),
                    "error": None,
                }

        except Exception as e:
            return {
                "prompt_id": prompt_data["id"],
                "category": prompt_data["category"],
                "severity": prompt_data["severity"],
                "attack_type": prompt_data["attack_type"],
                "technique": prompt_data["technique"],
                "model": MODELS.get(model_name, model_name),
                "turns": turns,
                "conversation_history": conversation_history,
                "response": None,
                "compliance_type": None,
                "finish_reason": None,
                "error": str(e),
            }

    # Should not reach here
    return {"error": "Multi-turn conversation failed unexpectedly"}


def run_jailbreak_tests(
    models: list[str],
    attack_type: str = None,
    test_mode: bool = False,
) -> list[dict[str, Any]]:
    """
    Run jailbreak tests across specified models.

    Args:
        models: List of model names to test
        attack_type: Specific attack type to test (None = all)
        test_mode: If True, only test 2 prompts per attack type

    Returns:
        List of all test results
    """
    prompts = load_jailbreak_prompts(attack_type)

    if test_mode:
        # In test mode, take 2 prompts per attack type
        attack_types = set(p["attack_type"] for p in prompts)
        test_prompts = []
        for at in attack_types:
            at_prompts = [p for p in prompts if p["attack_type"] == at]
            test_prompts.extend(at_prompts[:2])
        prompts = test_prompts

    total_tests = len(models) * len(prompts)
    print(f"\nRunning {total_tests} jailbreak tests:")
    print(f"  Models: {', '.join(models)}")
    print(f"  Prompts: {len(prompts)}")
    print(f"  Attack types: {', '.join(set(p['attack_type'] for p in prompts))}")
    print()

    all_results = []
    test_count = 0

    for model_name in models:
        model_display = MODELS.get(model_name, model_name)
        print(f"\n{'='*60}")
        print(f"Testing: {model_display}")
        print(f"{'='*60}\n")

        for prompt_data in prompts:
            test_count += 1
            print(f"[{test_count}/{total_tests}] {prompt_data['attack_type']}:{prompt_data['technique']} - {prompt_data['category']}", end=" ... ")

            # Handle multi-turn vs single-turn
            if prompt_data["attack_type"] == "multiturn":
                result = test_multi_turn(model_name, prompt_data)
            else:
                result = test_single_turn(model_name, prompt_data)

            all_results.append(result)

            # Print result
            if result["error"]:
                print(f"ERROR: {result['error']}")
            else:
                print(f"{result['compliance_type']}")

    return all_results


def print_summary(results: list[dict[str, Any]]) -> None:
    """Print summary statistics from jailbreak test results."""
    from collections import defaultdict

    # Calculate statistics per model
    models = defaultdict(lambda: {
        "total": 0,
        "refused": 0,
        "hedged": 0,
        "full": 0,
        "errors": 0,
    })

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

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}\n")

    for model in sorted(models.keys()):
        stats = models[model]
        total = stats["total"]

        if total == 0:
            print(f"{model}:")
            print(f"  No successful tests (errors: {stats['errors']})")
            print()
            continue

        print(f"{model}:")
        print(f"  Total prompts: {total}")
        print(f"  Refused: {stats['refused']} ({stats['refused']/total*100:.1f}%)")
        print(f"  Hedged: {stats['hedged']} ({stats['hedged']/total*100:.1f}%)")
        print(f"  Full compliance: {stats['full']} ({stats['full']/total*100:.1f}%)")
        if stats["errors"] > 0:
            print(f"  Errors: {stats['errors']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run jailbreak tests against local models"
    )
    parser.add_argument(
        "--models",
        type=str,
        default="local",
        help="Comma-separated model names or 'local' for all local models (default: local)"
    )
    parser.add_argument(
        "--attack-type",
        type=str,
        choices=["encoding", "roleplay", "injection", "multiturn"],
        help="Specific attack type to test (default: all)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode (2 prompts per attack type)"
    )

    args = parser.parse_args()

    # Parse model selection
    if args.models == "local":
        selected_models = ["llama3.2", "qwen3", "gemma3"]
    else:
        selected_models = [m.strip() for m in args.models.split(",")]

    # Run tests
    results = run_jailbreak_tests(
        models=selected_models,
        attack_type=args.attack_type,
        test_mode=args.test,
    )

    # Print summary
    print_summary(results)

    # Save results
    results_dir = SCRIPT_DIR / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = "_test" if args.test else ""
    output_file = results_dir / f"jailbreak_tests{mode_suffix}_{timestamp}.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
