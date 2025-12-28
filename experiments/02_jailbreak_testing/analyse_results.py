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
    ax.set_title('Baseline vs Jailbreak Success Rates', fontsize=14)
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
    ax.set_title('Jailbreak Success Rate by Attack Type', fontsize=14)
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

    ax.set_title("Model Vulnerability by Attack Type", fontsize=14)
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
    ax.set_title('Jailbreak Success Rate by Harm Category', fontsize=14)
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


def plot_radar_charts(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot radar charts with stacked compliance rings showing all compliance types."""
    models = sorted(df["model"].unique())
    categories = sorted(df["category"].unique())
    num_categories = len(categories)
    num_models = len(models)

    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create 2x2 polar subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                             subplot_kw=dict(projection='polar'))

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Colors for each compliance type
    colors = {
        "refused": "#2ecc71",   # Green
        "hedged": "#f39c12",    # Orange
        "full": "#e74c3c"       # Red
    }

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Calculate compliance rates for each category
        refused_rates = []
        hedged_rates = []
        full_rates = []

        for category in categories:
            subset = df[(df["model"] == model) & (df["category"] == category)]
            total = len(subset)

            if total > 0:
                refused = len(subset[subset["compliance_type"] == "refused"])
                hedged = len(subset[subset["compliance_type"] == "hedged"])
                full = len(subset[subset["compliance_type"] == "full"])

                # Calculate percentages
                refused_rates.append((refused / total) * 100)
                hedged_rates.append((hedged / total) * 100)
                full_rates.append((full / total) * 100)
            else:
                refused_rates.append(0)
                hedged_rates.append(0)
                full_rates.append(0)

        # Close the polygons
        refused_rates += refused_rates[:1]
        hedged_rates += hedged_rates[:1]
        full_rates += full_rates[:1]

        # Create stacked rings from center outward:
        # 1. Inner ring (0% to refused%): refused area (green) - safe at center
        # 2. Middle ring (refused% to refused%+hedged%): hedged area (orange)
        # 3. Outer ring (refused%+hedged% to 100%): full compliance (red) - dangerous at edge

        # Calculate cumulative percentages for stacking
        center = [0] * len(angles)
        refused_outer = refused_rates[:]
        hedged_outer = [(r + h) for r, h in zip(refused_rates, hedged_rates)]
        full_outer = [100] * len(angles)

        # Plot stacked areas from inside to outside
        ax.fill_between(angles, center, refused_outer,
                        alpha=0.7, color=colors["refused"], label='Refused')
        ax.fill_between(angles, refused_outer, hedged_outer,
                        alpha=0.7, color=colors["hedged"], label='Hedged')
        ax.fill_between(angles, hedged_outer, full_outer,
                        alpha=0.7, color=colors["full"], label='Full')

        # Add edge lines for clarity
        ax.plot(angles, refused_outer, 'o-', linewidth=1.5, color='#27ae60', alpha=0.8, markersize=4)
        ax.plot(angles, hedged_outer, 'o-', linewidth=1.5, color='#e67e22', alpha=0.8, markersize=4)

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=9.5)
        ax.grid(True, linestyle='-', alpha=0.5, linewidth=0.7, color='gray')

        # Title with compliance summary
        total_tests = len(df[df["model"] == model])
        refused_count = len(df[(df["model"] == model) & (df["compliance_type"] == "refused")])
        hedged_count = len(df[(df["model"] == model) & (df["compliance_type"] == "hedged")])
        full_count = len(df[(df["model"] == model) & (df["compliance_type"] == "full")])

        # Set model name as bold text at top
        ax.text(0.5, 1.17, model, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, fontweight='bold')

        # Add stats below the model name
        stats_text = f"R:{refused_count}/{total_tests} ({refused_count/total_tests*100:.1f}%) "
        stats_text += f"H:{hedged_count} ({hedged_count/total_tests*100:.1f}%) "
        stats_text += f"F:{full_count} ({full_count/total_tests*100:.1f}%)"
        ax.text(0.5, 1.12, stats_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9)

    # Hide unused subplots if fewer than 4 models
    for idx in range(num_models, 4):
        axes[idx].set_visible(False)

    # Add legend to the figure (top right, outside all plots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=11, frameon=True, title='Compliance Type')

    plt.suptitle("Jailbreak Attack Performance: Category-Specific Compliance Distribution",
                 fontsize=14, y=0.99, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    plt.savefig(output_dir / "radar_charts.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: radar_charts.png")


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
    plot_radar_charts(df, output_dir)
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
