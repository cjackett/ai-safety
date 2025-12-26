"""
Analyse and visualise jailbreak testing results.

Generates visualisations comparing baseline vs jailbreak success rates,
attack type effectiveness, and model vulnerability patterns.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Script directory
SCRIPT_DIR = Path(__file__).parent


def load_results(result_file: Path) -> list[dict[str, Any]]:
    """Load results from JSON file."""
    with open(result_file, "r") as f:
        return json.load(f)


def find_latest_results() -> Path:
    """Find the most recent results JSON file."""
    results_dir = SCRIPT_DIR / "results"

    if not results_dir.exists():
        raise FileNotFoundError("No results directory found")

    # Find all JSON files matching jailbreak pattern
    json_files = list(results_dir.glob("jailbreak_tests*.json"))

    if not json_files:
        raise FileNotFoundError("No results files found in results/")

    # Return the most recently modified file
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest


def create_jailbreak_summary(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create summary dataframe from jailbreak test results."""
    data = []
    for result in results:
        if result.get("error"):
            continue

        # Clean up model names (remove "ollama/" prefix)
        model_name = result["model"].replace("ollama/", "")

        data.append({
            "model": model_name,
            "category": result["category"],
            "attack_type": result["attack_type"],
            "technique": result["technique"],
            "compliance_type": result.get("compliance_type"),
            "severity": result.get("severity", "unknown"),
        })

    return pd.DataFrame(data)


def load_baseline_results() -> pd.DataFrame:
    """Load baseline results from 01_capability_probing for comparison."""
    baseline_dir = SCRIPT_DIR.parent / "01_capability_probing" / "results"

    # Find latest baseline results
    baseline_files = list(baseline_dir.glob("probe_models*.json"))
    if not baseline_files:
        return None

    latest_baseline = max(baseline_files, key=lambda p: p.stat().st_mtime)

    with open(latest_baseline, "r") as f:
        baseline_results = json.load(f)

    # Create summary
    data = []
    for result in baseline_results:
        if result.get("error"):
            continue

        model_name = result["model"].replace("ollama/", "")
        data.append({
            "model": model_name,
            "category": result["category"],
            "compliance_type": result.get("compliance_type"),
        })

    return pd.DataFrame(data)


def plot_baseline_vs_jailbreak_comparison(jailbreak_df: pd.DataFrame, baseline_df: pd.DataFrame, output_dir: Path) -> None:
    """Plot comparison of baseline vs jailbreak success rates."""
    if baseline_df is None:
        print("  ⚠ No baseline results found, skipping comparison plot")
        return

    models = sorted(jailbreak_df["model"].unique())

    # Calculate rates for each model
    baseline_rates = []
    jailbreak_rates = []

    for model in models:
        # Baseline full compliance rate
        baseline_subset = baseline_df[baseline_df["model"] == model]
        if len(baseline_subset) > 0:
            baseline_full = len(baseline_subset[baseline_subset["compliance_type"] == "full"])
            baseline_rate = (baseline_full / len(baseline_subset)) * 100
        else:
            baseline_rate = 0

        # Jailbreak full compliance rate
        jailbreak_subset = jailbreak_df[jailbreak_df["model"] == model]
        jailbreak_full = len(jailbreak_subset[jailbreak_subset["compliance_type"] == "full"])
        jailbreak_rate = (jailbreak_full / len(jailbreak_subset)) * 100

        baseline_rates.append(baseline_rate)
        jailbreak_rates.append(jailbreak_rate)

    # Create grouped bar chart
    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline (Direct Prompts)', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, jailbreak_rates, width, label='Jailbreak Attacks', color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Full Compliance Rate (%)', fontsize=12)
    ax.set_title('Baseline vs Jailbreak Success Rates', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.legend()
    ax.set_ylim(0, max(max(baseline_rates), max(jailbreak_rates)) + 10)

    # Add percentage labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%',
                       ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_vs_jailbreak.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: baseline_vs_jailbreak.png")


def plot_attack_type_effectiveness(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot success rates by attack type."""
    attack_types = sorted(df["attack_type"].unique())

    success_rates = []
    for attack_type in attack_types:
        subset = df[df["attack_type"] == attack_type]
        full_count = len(subset[subset["compliance_type"] == "full"])
        success_rate = (full_count / len(subset)) * 100
        success_rates.append(success_rate)

    # Create bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(attack_types, success_rates, color=['#e74c3c', '#f39c12', '#9b59b6', '#3498db'])

    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Jailbreak Success Rate (%)', fontsize=12)
    ax.set_title('Jailbreak Success Rate by Attack Type', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(success_rates) + 10)

    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}%',
               ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "attack_type_effectiveness.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: attack_type_effectiveness.png")


def plot_model_vulnerability_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot heatmap of jailbreak success rates by model and attack type."""
    models = sorted(df["model"].unique())
    attack_types = sorted(df["attack_type"].unique())

    # Calculate success rates
    success_rates = []
    for model in models:
        row = []
        for attack_type in attack_types:
            subset = df[(df["model"] == model) & (df["attack_type"] == attack_type)]
            if len(subset) > 0:
                full_count = len(subset[subset["compliance_type"] == "full"])
                rate = (full_count / len(subset)) * 100
                row.append(rate)
            else:
                row.append(0)
        success_rates.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 6))

    heatmap_df = pd.DataFrame(
        success_rates,
        index=models,
        columns=attack_types
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",  # Reversed: red=high vulnerability
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Jailbreak Success Rate (%)"},
        ax=ax
    )

    ax.set_title("Model Vulnerability by Attack Type", fontsize=14, fontweight='bold')
    ax.set_xlabel("Attack Type", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "vulnerability_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: vulnerability_heatmap.png")


def plot_category_jailbreak_rates(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot jailbreak success rates by harm category."""
    categories = sorted(df["category"].unique())

    success_rates = []
    for category in categories:
        subset = df[df["category"] == category]
        full_count = len(subset[subset["compliance_type"] == "full"])
        success_rate = (full_count / len(subset)) * 100
        success_rates.append(success_rate)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(categories, success_rates, color='#e74c3c', alpha=0.8)

    ax.set_xlabel('Jailbreak Success Rate (%)', fontsize=12)
    ax.set_ylabel('Harm Category', fontsize=12)
    ax.set_title('Jailbreak Success Rate by Harm Category', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(success_rates) + 10)

    # Add percentage labels
    for bar in bars:
        width = bar.get_width()
        if width > 0.5:
            ax.text(width + 1, bar.get_y() + bar.get_height()/2.,
                   f'{width:.1f}%',
                   ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "category_jailbreak_rates.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: category_jailbreak_rates.png")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse and visualise jailbreak testing results"
    )
    parser.add_argument(
        "result_file",
        type=str,
        nargs="?",
        default=None,
        help="Path to results JSON file (default: auto-detect latest)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as result file)"
    )

    args = parser.parse_args()

    # Auto-detect latest results file if not specified
    if args.result_file is None:
        try:
            result_file = find_latest_results()
            print(f"Auto-detected latest results file: {result_file.name}\n")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print("Please specify a results file explicitly.")
            return
    else:
        result_file = Path(args.result_file)
        if not result_file.exists():
            print(f"Error: Result file not found: {result_file}")
            return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = result_file.parent

    output_dir.mkdir(exist_ok=True)

    print(f"\nAnalysing results from: {result_file}")
    print(f"Output directory: {output_dir}\n")

    # Load and process results
    results = load_results(result_file)
    df = create_jailbreak_summary(results)

    # Load baseline results for comparison
    baseline_df = load_baseline_results()

    print(f"Loaded {len(df)} valid results\n")

    # Generate visualisations
    print("Generating visualisations:")
    plot_baseline_vs_jailbreak_comparison(df, baseline_df, output_dir)
    plot_attack_type_effectiveness(df, output_dir)
    plot_model_vulnerability_heatmap(df, output_dir)
    plot_category_jailbreak_rates(df, output_dir)
    generate_statistics_report(df, output_dir)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*60}\n")


def generate_statistics_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate detailed statistics report for jailbreak testing."""
    report = []
    report.append("# Jailbreak Testing Analysis Report")
    report.append(f"\nTotal jailbreak attempts: {len(df)}")
    report.append(f"Models tested: {df['model'].nunique()}")
    report.append(f"Attack types: {df['attack_type'].nunique()}")
    report.append(f"Harm categories: {df['category'].nunique()}")

    report.append("\n## Overall Compliance Distribution\n")
    compliance_overall = df["compliance_type"].value_counts()
    compliance_pct = (compliance_overall / len(df) * 100).round(2)
    for comp_type, count in compliance_overall.items():
        report.append(f"- {comp_type}: {count} ({compliance_pct.get(comp_type, 0)}%)")

    report.append("\n## Attack Type Effectiveness (Full Compliance Rate)\n")
    attack_stats = df.groupby("attack_type")["compliance_type"].apply(
        lambda x: (x == "full").sum() / len(x) * 100
    ).sort_values(ascending=False)
    for attack, rate in attack_stats.items():
        report.append(f"- {attack}: {rate:.1f}% success rate")

    report.append("\n## Model Vulnerability (Full Compliance Rate)\n")
    for model in sorted(df["model"].unique()):
        model_df = df[df["model"] == model]
        full_count = (model_df["compliance_type"] == "full").sum()
        hedged_count = (model_df["compliance_type"] == "hedged").sum()
        total = len(model_df)
        report.append(f"\n### {model}")
        report.append(f"- Jailbreak Success (Full): {full_count} ({full_count/total*100:.1f}%)")
        report.append(f"- Partial Success (Hedged): {hedged_count} ({hedged_count/total*100:.1f}%)")
        report.append(f"- Refused: {total - full_count - hedged_count} ({(total - full_count - hedged_count)/total*100:.1f}%)")

    # Save report
    with open(output_dir / "statistics_report.md", "w") as f:
        f.write("\n".join(report))

    print(f"  ✓ Saved: statistics_report.md")


if __name__ == "__main__":
    main()
