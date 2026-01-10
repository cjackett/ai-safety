# ruff: noqa: INP001
"""
Analyse and visualise capability probing results.

Generates various plots to understand model compliance patterns across
harm categories and prompt types.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.container import BarContainer

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_results(result_file: Path) -> list[dict[str, Any]]:
    """Load results from JSON file."""
    with result_file.open() as f:
        return cast("list[dict[str, Any]]", json.load(f))


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

        data.append(
            {
                "model": model_name,
                "category": result["category"],
                "prompt_id": result["prompt_id"],
                "compliance_type": compliance_type,
                "severity": result.get("severity", "unknown"),
            },
        )

    return pd.DataFrame(data)


def _category_compliance_rates(
    df: pd.DataFrame,
    model: str,
    categories: list[str],
) -> tuple[list[float], list[float], list[float]]:
    refused_rates: list[float] = []
    hedged_rates: list[float] = []
    full_rates: list[float] = []

    for category in categories:
        subset = df[(df["model"] == model) & (df["category"] == category)]
        total = len(subset)

        if total > 0:
            refused = len(subset[subset["compliance_type"] == "refused"])
            hedged = len(subset[subset["compliance_type"] == "hedged"])
            full = len(subset[subset["compliance_type"] == "full"])

            refused_rates.append((refused / total) * 100)
            hedged_rates.append((hedged / total) * 100)
            full_rates.append((full / total) * 100)
        else:
            refused_rates.append(0)
            hedged_rates.append(0)
            full_rates.append(0)

    return refused_rates, hedged_rates, full_rates


def _annotate_radar_stats(
    ax: Axes,
    model: str,
    total_tests: int,
    refused_count: int,
    hedged_count: int,
    full_count: int,
) -> None:
    ax.text(0.5, 1.17, model, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")

    stats_text = f"R:{refused_count}/{total_tests} ({refused_count/total_tests*100:.1f}%) "
    stats_text += f"H:{hedged_count} ({hedged_count/total_tests*100:.1f}%) "
    stats_text += f"F:{full_count} ({full_count/total_tests*100:.1f}%)"
    ax.text(0.5, 1.12, stats_text, transform=ax.transAxes, ha="center", va="top", fontsize=9)


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
    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"projection": "polar"})

    # Flatten axes for easy iteration
    axes = axes.flatten()

    # Colors for each compliance type
    colors = {"refused": "#2ecc71", "hedged": "#f39c12", "full": "#e74c3c"}  # Green  # Orange  # Red

    for idx, model in enumerate(models):
        ax = axes[idx]

        refused_rates, hedged_rates, full_rates = _category_compliance_rates(df, model, categories)

        # Close the polygons
        refused_rates += refused_rates[:1]
        hedged_rates += hedged_rates[:1]
        full_rates += full_rates[:1]

        # Create stacked rings from center outward:
        # 1. Inner ring (0% to refused%): refused area (green) - safe at center
        # 2. Middle ring (refused% to refused%+hedged%): hedged area (orange)
        # 3. Outer ring (refused%+hedged% to 100%): full compliance (red) - dangerous at edge

        # Calculate cumulative percentages for stacking
        # Start from center (0%) and work outward
        center = [0] * len(angles)

        # Refused region: from 0% to refused%
        refused_outer = refused_rates[:]

        # Hedged region: from refused% to (refused% + hedged%)
        hedged_outer = [(r + h) for r, h in zip(refused_rates, hedged_rates, strict=False)]

        # Full region: from (refused% + hedged%) to (refused% + hedged% + full%)
        full_outer = [(r + h + f) for r, h, f in zip(refused_rates, hedged_rates, full_rates, strict=False)]

        # Plot stacked areas from inside to outside
        # 1. Refused (innermost green ring) - safe at center
        ax.fill_between(angles, center, refused_outer, alpha=0.7, color=colors["refused"], label="Refused")

        # 2. Hedged (middle orange ring)
        ax.fill_between(angles, refused_outer, hedged_outer, alpha=0.7, color=colors["hedged"], label="Hedged")

        # 3. Full compliance (outermost red ring) - dangerous at edge
        ax.fill_between(angles, hedged_outer, full_outer, alpha=0.7, color=colors["full"], label="Full")

        # Add edge lines for clarity
        ax.plot(angles, refused_outer, "o-", linewidth=1.5, color="#27ae60", alpha=0.8, markersize=4)
        ax.plot(angles, hedged_outer, "o-", linewidth=1.5, color="#e67e22", alpha=0.8, markersize=4)

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=9.5)
        ax.grid(True, linestyle="-", alpha=0.5, linewidth=0.7, color="gray")

        # Title with compliance summary
        total_tests = len(df[df["model"] == model])
        refused_count = len(df[(df["model"] == model) & (df["compliance_type"] == "refused")])
        hedged_count = len(df[(df["model"] == model) & (df["compliance_type"] == "hedged")])
        full_count = len(df[(df["model"] == model) & (df["compliance_type"] == "full")])

        _annotate_radar_stats(
            ax,
            model,
            total_tests,
            refused_count,
            hedged_count,
            full_count,
        )

    # Hide unused subplots if fewer than 4 models
    for idx in range(num_models, 4):
        axes[idx].set_visible(False)

    # Add legend to the figure (top right, outside all plots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=11,
        frameon=True,
        title="Compliance Type",
    )

    plt.suptitle(
        "Model Safety Performance: Category-Specific Compliance Distribution",
        fontsize=14,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 0.88, 0.97))
    plt.savefig(output_dir / "radar_charts.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("  ✓ Saved: radar_charts.png")


def plot_compliance_by_model(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot compliance type breakdown by model."""
    # Count compliance types per model
    compliance_counts = df.pivot_table(
        index="model",
        columns="compliance_type",
        aggfunc="size",
        fill_value=0,
    )

    # Calculate percentages
    compliance_pct = compliance_counts.div(compliance_counts.sum(axis=1), axis=0) * 100

    # Define colors for each compliance type (use gray for unknown)
    colors = {"refused": "#2ecc71", "hedged": "#f39c12", "full": "#e74c3c", "unknown": "#95a5a6"}

    # Only use colors that exist in the data
    plot_colors = {k: v for k, v in colors.items() if k in compliance_pct.columns}

    # Create vertical stacked bar chart
    _fig, ax = plt.subplots(figsize=(10, 6))
    compliance_pct.plot(kind="bar", stacked=True, ax=ax, color=plot_colors)

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Compliance Type Distribution by Model", fontsize=14)
    ax.legend(title="Compliance Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_ylim(0, 100)

    # Rotate x-axis labels to be horizontal (0 degrees)
    plt.xticks(rotation=0, ha="center")

    # Add percentage labels, but skip 0.0% to avoid clutter
    label_threshold = 0.5
    for container in ax.containers:
        if isinstance(container, BarContainer):
            values = cast("list[float]", getattr(container, "datavalues", []))
            labels = [f"{v:.1f}%" if v > label_threshold else "" for v in values]
            ax.bar_label(container, labels=labels, label_type="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / "compliance_by_model.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("  ✓ Saved: compliance_by_model.png")


def plot_compliance_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot compliance type breakdown by harm category."""
    # Count compliance types per category
    compliance_counts = df.pivot_table(
        index="category",
        columns="compliance_type",
        aggfunc="size",
        fill_value=0,
    )

    # Calculate percentages
    compliance_pct = compliance_counts.div(compliance_counts.sum(axis=1), axis=0) * 100

    # Define colors for each compliance type (use gray for unknown)
    colors = {"refused": "#2ecc71", "hedged": "#f39c12", "full": "#e74c3c", "unknown": "#95a5a6"}

    # Only use colors that exist in the data
    plot_colors = {k: v for k, v in colors.items() if k in compliance_pct.columns}

    # Create stacked bar chart
    _fig, ax = plt.subplots(figsize=(12, 8))
    compliance_pct.plot(kind="barh", stacked=True, ax=ax, color=plot_colors)

    ax.set_xlabel("Percentage (%)", fontsize=12)
    ax.set_ylabel("Harm Category", fontsize=12)
    ax.set_title("Compliance Type Distribution by Category", fontsize=14)
    ax.legend(title="Compliance Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "compliance_by_category.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("  ✓ Saved: compliance_by_category.png")


def plot_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot heatmap of refusal rates (model vs category)."""
    # Calculate refusal rate per model-category combination
    df.pivot_table(
        index=["model", "category"],
        columns="compliance_type",
        aggfunc="size",
        fill_value=0,
    )

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
    _fig, ax = plt.subplots(figsize=(14, 6))

    heatmap_df = pd.DataFrame(refusal_rates, index=df["model"].unique(), columns=df["category"].unique())

    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Refusal Rate (%)"},
        ax=ax,
    )

    ax.set_title("Refusal Rate Heatmap (Model x Category)", fontsize=14)
    ax.set_xlabel("Harm Category", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)

    # Rotate x-axis labels 45 degrees
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig(output_dir / "refusal_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("  ✓ Saved: refusal_heatmap.png")


def plot_hedged_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze hedged compliance patterns."""
    hedged = df[df["compliance_type"] == "hedged"]

    if len(hedged) == 0:
        logger.warning("  ⚠ No hedged responses found, skipping hedged analysis")
        return

    # Count hedged responses by model and category
    _fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # By model
    hedged_by_model = hedged.groupby("model").size().sort_values(ascending=True)
    hedged_by_model.plot(kind="barh", ax=ax1, color="#f39c12")
    ax1.set_xlabel("Number of Hedged Responses", fontsize=12)
    ax1.set_ylabel("Model", fontsize=12)
    ax1.set_title("Hedged Compliance by Model", fontsize=14)

    # By category
    hedged_by_category = hedged.groupby("category").size().sort_values(ascending=True)
    hedged_by_category.plot(kind="barh", ax=ax2, color="#e67e22")
    ax2.set_xlabel("Number of Hedged Responses", fontsize=12)
    ax2.set_ylabel("Harm Category", fontsize=12)
    ax2.set_title("Hedged Compliance by Category", fontsize=14)

    plt.tight_layout()
    plt.savefig(output_dir / "hedged_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("  ✓ Saved: hedged_analysis.png")


def plot_severity_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Analyze compliance by prompt severity."""
    if "severity" not in df.columns or df["severity"].isna().all():
        logger.warning("  ⚠ No severity data found, skipping severity analysis")
        return

    # Count compliance types by severity
    severity_compliance = df.pivot_table(
        index="severity",
        columns="compliance_type",
        aggfunc="size",
        fill_value=0,
    )
    severity_pct = severity_compliance.div(severity_compliance.sum(axis=1), axis=0) * 100

    # Define colors for each compliance type (use gray for unknown)
    colors = {"refused": "#2ecc71", "hedged": "#f39c12", "full": "#e74c3c", "unknown": "#95a5a6"}

    # Only use colors that exist in the data
    plot_colors = {k: v for k, v in colors.items() if k in severity_pct.columns}

    _fig, ax = plt.subplots(figsize=(10, 6))
    severity_pct.plot(kind="bar", stacked=True, ax=ax, color=plot_colors)

    ax.set_xlabel("Prompt Severity", fontsize=12)
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Compliance Type Distribution by Severity", fontsize=14)
    ax.legend(title="Compliance Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(output_dir / "compliance_by_severity.png", dpi=300, bbox_inches="tight")
    plt.close()

    logger.info("  ✓ Saved: compliance_by_severity.png")


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
        cat_by_model = cat_df.pivot_table(
            index="model",
            columns="compliance_type",
            aggfunc="size",
            fill_value=0,
        )
        if "full" in cat_by_model.columns:
            most_vulnerable = cat_by_model["full"].idxmax()
            if cat_by_model.loc[most_vulnerable, "full"] > 0:
                report.append(f"- Most vulnerable model: {most_vulnerable}")

    # Save report
    with (output_dir / "statistics_report.md").open("w") as f:
        f.write("\n".join(report))

    logger.info("  ✓ Saved: statistics_report.md")


def find_latest_results() -> Path:
    """Find the most recent results JSON file in results/raw/ directory."""
    script_dir = Path(__file__).parent
    results_dir = script_dir / "results" / "raw"

    if not results_dir.exists():
        message = f"No results/raw/ directory found at: {results_dir}"
        raise FileNotFoundError(message)

    # Find all JSON files matching probe_models pattern
    json_files = list(results_dir.glob("probe_models*.json"))

    if not json_files:
        message = "No results files found in results/raw/"
        raise FileNotFoundError(message)

    # Return the most recently modified file
    return max(json_files, key=lambda p: p.stat().st_mtime)


def main() -> None:
    """Run analysis for the latest or specified results file."""
    parser = argparse.ArgumentParser(description="Analyse and visualise capability probing results")
    parser.add_argument(
        "result_file",
        type=str,
        nargs="?",  # Make it optional
        default=None,
        help="Path to results JSON file (default: auto-detect latest)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (default: same as result file)",
    )

    args = parser.parse_args()

    # Auto-detect latest results file if not specified
    if args.result_file is None:
        try:
            result_file = find_latest_results()
            logger.info("Auto-detected latest results file: %s", result_file.name)
        except FileNotFoundError:
            logger.exception("Error finding results file.")
            logger.info("Please specify a results file explicitly.")
            return
    else:
        result_file = Path(args.result_file)
        if not result_file.exists():
            logger.error("Result file not found: %s", result_file)
            return

    # Create standard directory structure
    script_dir = Path(__file__).parent
    figures_dir = script_dir / "results" / "figures"
    analysis_dir = script_dir / "results" / "analysis"

    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Override output_dir if specified
    if args.output_dir:
        figures_dir = Path(args.output_dir)
        figures_dir.mkdir(exist_ok=True)

    logger.info("")
    logger.info("Analyzing results from: %s", result_file)
    logger.info("Figures will be saved to: %s", figures_dir)
    logger.info("Analysis will be saved to: %s", analysis_dir)

    # Load and process results
    results = load_results(result_file)
    df = create_compliance_summary(results)

    logger.info("")
    logger.info("Loaded %d valid results", len(df))

    # Generate visualizations
    logger.info("")
    logger.info("Generating visualizations:")
    plot_radar_charts(df, figures_dir)
    plot_compliance_by_model(df, figures_dir)
    plot_heatmap(df, figures_dir)
    generate_statistics_report(df, analysis_dir)

    logger.info("")
    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("Figures saved to: %s", figures_dir)
    logger.info("Analysis saved to: %s", analysis_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
