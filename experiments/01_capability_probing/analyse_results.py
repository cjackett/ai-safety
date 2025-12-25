"""
Analyse and visualise capability probing results.

Generates various plots to understand model compliance patterns across
harm categories and prompt types.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_results(result_file: Path) -> list[dict[str, Any]]:
    """Load results from JSON file."""
    with open(result_file, "r") as f:
        return json.load(f)


def create_compliance_summary(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Create summary dataframe from results."""
    data = []
    for result in results:
        if result["error"]:
            continue

        # Handle backwards compatibility with old "refused" field
        if "compliance_type" in result:
            compliance_type = result["compliance_type"]
        elif "refused" in result:
            # Convert old boolean refused field to compliance_type
            if result["refused"] is True:
                compliance_type = "refused"
            elif result["refused"] is False:
                compliance_type = "full"
            else:
                compliance_type = "unknown"
        else:
            compliance_type = "unknown"

        # Clean up model names (remove "ollama/" prefix)
        model_name = result["model"].replace("ollama/", "")

        data.append({
            "model": model_name,
            "category": result["category"],
            "prompt_id": result["prompt_id"],
            "compliance_type": compliance_type,
            "severity": result.get("severity", "unknown"),
        })

    return pd.DataFrame(data)


def plot_radar_charts(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot radar charts showing refusal rates across categories for each model."""
    import numpy as np

    models = df["model"].unique()
    categories = sorted(df["category"].unique())
    num_categories = len(categories)

    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create subplot grid
    num_models = len(models)
    cols = min(3, num_models)
    rows = (num_models + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), subplot_kw=dict(projection='polar'))

    # Flatten axes array for easy iteration
    if num_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if num_models > 1 else [axes]

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Calculate refusal rate for each category
        refusal_rates = []
        for category in categories:
            subset = df[(df["model"] == model) & (df["category"] == category)]
            if len(subset) > 0:
                refused = len(subset[subset["compliance_type"] == "refused"])
                rate = (refused / len(subset)) * 100
                refusal_rates.append(rate)
            else:
                refusal_rates.append(0)

        # Close the plot
        refusal_rates += refusal_rates[:1]

        # Plot
        ax.plot(angles, refusal_rates, 'o-', linewidth=2, color='#2ecc71')
        ax.fill(angles, refusal_rates, alpha=0.25, color='#2ecc71')

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=8)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Title
        ax.set_title(f"{model}", size=14, fontweight='bold', pad=20)

    # Hide unused subplots
    for idx in range(num_models, len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle("Model Refusal Rates Across Harm Categories",
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])  # Leave space at top for title
    plt.savefig(output_dir / "radar_charts.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: radar_charts.png")


def plot_compliance_by_model(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot compliance type breakdown by model."""
    # Count compliance types per model
    compliance_counts = df.groupby(["model", "compliance_type"]).size().unstack(fill_value=0)

    # Calculate percentages
    compliance_pct = compliance_counts.div(compliance_counts.sum(axis=1), axis=0) * 100

    # Define colors for each compliance type (use gray for unknown)
    colors = {"refused": "#2ecc71", "hedged": "#f39c12", "full": "#e74c3c", "unknown": "#95a5a6"}

    # Only use colors that exist in the data
    plot_colors = {k: v for k, v in colors.items() if k in compliance_pct.columns}

    # Create vertical stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    compliance_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=plot_colors
    )

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Compliance Type Distribution by Model", fontsize=14, fontweight="bold")
    ax.legend(title="Compliance Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 100)

    # Rotate x-axis labels to be horizontal (0 degrees)
    plt.xticks(rotation=0, ha='center')

    # Add percentage labels, but skip 0.0% to avoid clutter
    for container in ax.containers:
        labels = [f"{v:.1f}%" if v > 0.5 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "compliance_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: compliance_by_model.png")


def plot_compliance_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot compliance type breakdown by harm category."""
    # Count compliance types per category
    compliance_counts = df.groupby(["category", "compliance_type"]).size().unstack(fill_value=0)

    # Calculate percentages
    compliance_pct = compliance_counts.div(compliance_counts.sum(axis=1), axis=0) * 100

    # Define colors for each compliance type (use gray for unknown)
    colors = {"refused": "#2ecc71", "hedged": "#f39c12", "full": "#e74c3c", "unknown": "#95a5a6"}

    # Only use colors that exist in the data
    plot_colors = {k: v for k, v in colors.items() if k in compliance_pct.columns}

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    compliance_pct.plot(
        kind="barh",
        stacked=True,
        ax=ax,
        color=plot_colors
    )

    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("Harm Category", fontsize=12)
    ax.set_title("Compliance Type Distribution by Category", fontsize=14, fontweight="bold")
    ax.legend(title="Compliance Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "compliance_by_category.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: compliance_by_category.png")


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot heatmap of refusal rates (model vs category)."""
    # Calculate refusal rate per model-category combination
    pivot = df.groupby(["model", "category", "compliance_type"]).size().unstack(fill_value=0)

    # Calculate refusal percentage (refused / total)
    refusal_rates = []
    for model in df["model"].unique():
        row = []
        for category in df["category"].unique():
            subset = df[(df["model"] == model) & (df["category"] == category)]
            if len(subset) > 0:
                refused = len(subset[subset["compliance_type"] == "refused"])
                rate = (refused / len(subset)) * 100
                row.append(rate)
            else:
                row.append(0)
        refusal_rates.append(row)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    heatmap_df = pd.DataFrame(
        refusal_rates,
        index=df["model"].unique(),
        columns=df["category"].unique()
    )

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Refusal Rate (%)"},
        ax=ax
    )

    ax.set_title("Refusal Rate Heatmap (Model × Category)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Harm Category", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    # Rotate x-axis labels 45 degrees
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(output_dir / "refusal_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: refusal_heatmap.png")


def plot_hedged_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze hedged compliance patterns."""
    hedged = df[df["compliance_type"] == "hedged"]

    if len(hedged) == 0:
        print("  ⚠ No hedged responses found, skipping hedged analysis")
        return

    # Count hedged responses by model and category
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # By model
    hedged_by_model = hedged.groupby("model").size().sort_values(ascending=True)
    hedged_by_model.plot(kind="barh", ax=ax1, color="#f39c12")
    ax1.set_xlabel("Number of Hedged Responses", fontsize=12)
    ax1.set_ylabel("Model", fontsize=12)
    ax1.set_title("Hedged Compliance by Model", fontsize=14, fontweight="bold")

    # By category
    hedged_by_category = hedged.groupby("category").size().sort_values(ascending=True)
    hedged_by_category.plot(kind="barh", ax=ax2, color="#e67e22")
    ax2.set_xlabel("Number of Hedged Responses", fontsize=12)
    ax2.set_ylabel("Harm Category", fontsize=12)
    ax2.set_title("Hedged Compliance by Category", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "hedged_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: hedged_analysis.png")


def plot_severity_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze compliance by prompt severity."""
    if "severity" not in df.columns or df["severity"].isna().all():
        print("  ⚠ No severity data found, skipping severity analysis")
        return

    # Count compliance types by severity
    severity_compliance = df.groupby(["severity", "compliance_type"]).size().unstack(fill_value=0)
    severity_pct = severity_compliance.div(severity_compliance.sum(axis=1), axis=0) * 100

    # Define colors for each compliance type (use gray for unknown)
    colors = {"refused": "#2ecc71", "hedged": "#f39c12", "full": "#e74c3c", "unknown": "#95a5a6"}

    # Only use colors that exist in the data
    plot_colors = {k: v for k, v in colors.items() if k in severity_pct.columns}

    fig, ax = plt.subplots(figsize=(10, 6))
    severity_pct.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=plot_colors
    )

    ax.set_xlabel("Prompt Severity", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Compliance Type Distribution by Severity", fontsize=14, fontweight="bold")
    ax.legend(title="Compliance Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "compliance_by_severity.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: compliance_by_severity.png")


def generate_statistics_report(df: pd.DataFrame, output_dir: Path) -> None:
    """Generate detailed statistics report."""
    report = []
    report.append("# Capability Probing Analysis Report")
    report.append(f"\nTotal prompts analysed: {len(df)}")
    report.append(f"Models tested: {df['model'].nunique()}")
    report.append(f"Harm categories: {df['category'].nunique()}")

    report.append("\n## Overall Compliance Distribution\n")
    compliance_overall = df["compliance_type"].value_counts()
    compliance_pct = (compliance_overall / len(df) * 100).round(2)
    for comp_type, count in compliance_overall.items():
        report.append(f"- {comp_type}: {count} ({compliance_pct[comp_type]}%)")

    report.append("\n## Model Performance\n")
    for model in df["model"].unique():
        model_df = df[df["model"] == model]
        refused = len(model_df[model_df["compliance_type"] == "refused"])
        hedged = len(model_df[model_df["compliance_type"] == "hedged"])
        full = len(model_df[model_df["compliance_type"] == "full"])
        total = len(model_df)

        report.append(f"\n### {model}")
        report.append(f"- Total: {total}")
        report.append(f"- Refused: {refused} ({refused/total*100:.1f}%)")
        report.append(f"- Hedged: {hedged} ({hedged/total*100:.1f}%)")
        report.append(f"- Full: {full} ({full/total*100:.1f}%)")

    report.append("\n## Category Analysis\n")
    for category in sorted(df["category"].unique()):
        cat_df = df[df["category"] == category]
        refused = len(cat_df[cat_df["compliance_type"] == "refused"])
        total = len(cat_df)

        report.append(f"\n### {category}")
        report.append(f"- Refusal rate: {refused/total*100:.1f}%")

        # Most vulnerable model for this category
        cat_by_model = cat_df.groupby(["model", "compliance_type"]).size().unstack(fill_value=0)
        if "full" in cat_by_model.columns:
            most_vulnerable = cat_by_model["full"].idxmax()
            if cat_by_model.loc[most_vulnerable, "full"] > 0:
                report.append(f"- Most vulnerable model: {most_vulnerable}")

    # Save report
    with open(output_dir / "statistics_report.md", "w") as f:
        f.write("\n".join(report))

    print(f"  ✓ Saved: statistics_report.md")


def find_latest_results() -> Path:
    """Find the most recent results JSON file."""
    # Look for results in the default results directory
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results"

    if not results_dir.exists():
        raise FileNotFoundError("No results directory found")

    # Find all JSON files matching probe_models pattern
    json_files = list(results_dir.glob("probe_models*.json"))

    if not json_files:
        raise FileNotFoundError("No results files found in results/")

    # Return the most recently modified file
    latest = max(json_files, key=lambda p: p.stat().st_mtime)
    return latest


def main():
    parser = argparse.ArgumentParser(
        description="Analyse and visualise capability probing results"
    )
    parser.add_argument(
        "result_file",
        type=str,
        nargs="?",  # Make it optional
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
        output_dir = result_file.parent  # Save directly to results/ directory

    output_dir.mkdir(exist_ok=True)

    print(f"\nAnalyzing results from: {result_file}")
    print(f"Output directory: {output_dir}\n")

    # Load and process results
    results = load_results(result_file)
    df = create_compliance_summary(results)

    print(f"Loaded {len(df)} valid results\n")

    # Generate visualisations
    print("Generating visualisations:")
    plot_radar_charts(df, output_dir)
    plot_compliance_by_model(df, output_dir)
    plot_heatmap(df, output_dir)
    generate_statistics_report(df, output_dir)

    print(f"\n{'='*60}")
    print(f"Analysis complete! Plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
