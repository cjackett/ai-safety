#!/usr/bin/env python3
"""
Run all Inspect AI evaluations across all models.

This script runs all 3 tasks (CBRN, Cyber, Persuasion) on all 4 models,
using JSON log format for easy parsing and analysis.

Can be run from any directory - it will automatically find the experiment directory.

Usage:
    python experiments/08_inspect_ai_evaluations/run_inspect_evals.py              # Run all
    python experiments/08_inspect_ai_evaluations/run_inspect_evals.py --test       # Test mode
    python experiments/08_inspect_ai_evaluations/run_inspect_evals.py --models llama3.2:3b qwen3:4b
"""

import subprocess
import argparse
from pathlib import Path
import sys
import os

# Models to evaluate (from Experiment 01)
DEFAULT_MODELS = [
    "llama3.2:3b",
    "qwen3:4b",
    "gemma3:4b",
    "mistral:7b"
]

# Tasks to run
TASKS = [
    "cbrn_capability",
    "cyber_offense",
    "persuasion"
]

def run_evaluation(task: str, model: str, script_dir: Path, limit: int = None):
    """Run a single Inspect AI evaluation."""
    task_file_abs = script_dir / "tasks" / f"{task}.py"
    task_file_rel = f"tasks/{task}.py"  # Relative path for Inspect

    if not task_file_abs.exists():
        print(f"❌ Error: Task file not found: {task_file_abs}")
        return False

    cmd = [
        "inspect", "eval",
        task_file_rel,  # Use relative path - Inspect doesn't support absolute paths
        "--model", f"ollama/{model}",
        "--log-format", "json",  # Use JSON format instead of binary .eval
    ]

    if limit:
        cmd.extend(["--limit", str(limit)])

    print(f"\n{'='*80}")
    print(f"Running: {task} on {model}")
    if limit:
        print(f"(Limited to {limit} samples for testing)")
    print(f"{'='*80}\n")

    try:
        # Run from the script directory
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
            text=True,
            cwd=script_dir  # Run from experiment directory
        )
        print(f"\n✅ Completed: {task} on {model}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed: {task} on {model}")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠️  Interrupted: {task} on {model}")
        raise

def main():
    parser = argparse.ArgumentParser(
        description="Run Inspect AI evaluations across all tasks and models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help=f"Models to evaluate (default: {', '.join(DEFAULT_MODELS)})"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        help=f"Tasks to run (default: {', '.join(TASKS)})"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: limit to 5 samples per evaluation"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of samples per evaluation"
    )

    args = parser.parse_args()

    # Get script directory (where this script is located)
    script_dir = Path(__file__).parent.resolve()

    # Verify we're in the right place
    if not (script_dir / "tasks").exists():
        print(f"❌ Error: Could not find 'tasks' directory in {script_dir}")
        print(f"   Please run this script from the experiment directory:")
        print(f"   cd {script_dir}")
        print(f"   python run_inspect_evals.py")
        sys.exit(1)

    # Determine sample limit
    limit = None
    if args.test:
        limit = 5
    elif args.limit:
        limit = args.limit

    # Summary
    total_evals = len(args.tasks) * len(args.models)
    print(f"\n{'='*80}")
    print(f"Inspect AI Evaluation Suite")
    print(f"{'='*80}")
    print(f"Working directory: {script_dir}")
    print(f"Tasks: {', '.join(args.tasks)}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Total evaluations: {total_evals}")
    if limit:
        print(f"Sample limit: {limit}")
    print(f"Log format: JSON")
    print(f"{'='*80}\n")

    # Run evaluations
    results = []
    total = 0
    succeeded = 0
    failed = 0

    try:
        for task in args.tasks:
            for model in args.models:
                total += 1
                success = run_evaluation(task, model, script_dir, limit)
                results.append({
                    'task': task,
                    'model': model,
                    'success': success
                })
                if success:
                    succeeded += 1
                else:
                    failed += 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)

    # Summary report
    print(f"\n\n{'='*80}")
    print(f"Evaluation Summary")
    print(f"{'='*80}")
    print(f"Total evaluations: {total}")
    print(f"✅ Succeeded: {succeeded}")
    print(f"❌ Failed: {failed}")

    if failed > 0:
        print(f"\nFailed evaluations:")
        for result in results:
            if not result['success']:
                print(f"  - {result['task']} on {result['model']}")

    print(f"\nLogs saved to: {script_dir / 'logs'}/")
    print(f"Format: JSON (use analyse_results.py to generate visualizations)")
    print(f"{'='*80}\n")

    sys.exit(0 if failed == 0 else 1)

if __name__ == "__main__":
    main()
