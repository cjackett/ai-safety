# ruff: noqa: INP001
"""Analysis and visualization for behavioral safety evaluation results.

Generates:
- Alignment distribution charts by model
- Category-specific heatmaps
- Behavioral concern breakdown
- Radar charts showing performance across categories
- Detailed statistics report

Reads from: results/raw/
Saves to: results/figures/ and results/analysis/
"""


import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, TextIO, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

# Script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

# Logger
logger = logging.getLogger(__name__)

MIN_LABEL_PERCENT = 3.0
RADAR_MAX_PERCENT = 100


def setup_logging() -> None:
    """Setup basic logging for analysis script."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = SCRIPT_DIR / "config" / "config.yaml"

    with config_path.open() as f:
        return cast("dict[str, Any]", yaml.safe_load(f))


def load_results(results_file: Path | None = None) -> list[dict[str, Any]]:
    """Load behavioral evaluation results from JSON file."""
    if results_file is None:
        # Find most recent results file in raw/ directory
        results_dir = SCRIPT_DIR / "results" / "raw"

        if not results_dir.exists():
            msg = f"Results directory not found: {results_dir}. Run run_behavioral_evals.py first to generate results."
            raise FileNotFoundError(msg)

        json_files = list(results_dir.glob("behavioral_eval*.json"))

        if not json_files:
            msg = f"No results files found in {results_dir}. Run run_behavioral_evals.py first to generate results."
            raise FileNotFoundError(msg)

        results_file = max(json_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Auto-detected results file: {results_file.name}")

    with results_file.open() as f:
        return cast("list[dict[str, Any]]", json.load(f))


def compute_alignment_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Compute overall alignment statistics by model."""
    model_stats: defaultdict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0,
            "aligned": 0,
            "concerning": 0,
            "problematic": 0,
        },
    )

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        alignment = result.get("behavioral_alignment")

        # Only count results with valid classifications
        if alignment == "aligned":
            model_stats[model]["total"] += 1
            model_stats[model]["aligned"] += 1
        elif alignment == "concerning":
            model_stats[model]["total"] += 1
            model_stats[model]["concerning"] += 1
        elif alignment == "problematic":
            model_stats[model]["total"] += 1
            model_stats[model]["problematic"] += 1

    return dict(model_stats)


def compute_category_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, int]]]:
    """Compute alignment statistics by category and model."""
    category_stats: defaultdict[str, defaultdict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(
            lambda: {
                "aligned": 0,
                "concerning": 0,
                "problematic": 0,
                "total": 0,
            },
        ),
    )

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        category = result.get("category", "unknown")
        alignment = result.get("behavioral_alignment")

        # Only count results with valid classifications
        if alignment == "aligned":
            category_stats[category][model]["total"] += 1
            category_stats[category][model]["aligned"] += 1
        elif alignment == "concerning":
            category_stats[category][model]["total"] += 1
            category_stats[category][model]["concerning"] += 1
        elif alignment == "problematic":
            category_stats[category][model]["total"] += 1
            category_stats[category][model]["problematic"] += 1

    return {category: dict(model_stats) for category, model_stats in category_stats.items()}


def compute_flag_stats(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Compute behavioral flag statistics."""
    flag_stats: defaultdict[str, defaultdict[str, int]] = defaultdict(lambda: defaultdict(int))

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        flags = result.get("behavioral_flags", [])

        for flag in flags:
            flag_stats[model][flag] += 1

    return {model: dict(stats) for model, stats in flag_stats.items()}


def plot_alignment_distribution(model_stats: dict[str, dict[str, int]], output_dir: Path) -> None:
    """Create stacked bar chart of alignment distribution by model."""
    models = sorted(model_stats.keys())
    aligned = []
    concerning = []
    problematic = []

    for model in models:
        stats = model_stats[model]
        total = stats["total"]

        if total > 0:
            aligned.append((stats["aligned"] / total) * 100)
            concerning.append((stats["concerning"] / total) * 100)
            problematic.append((stats["problematic"] / total) * 100)
        else:
            aligned.append(0)
            concerning.append(0)
            problematic.append(0)

    # Create plot
    _fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.6

    # Stack bars
    ax.bar(x, aligned, width, label="Aligned", color="#2ecc71")
    ax.bar(x, concerning, width, bottom=aligned, label="Concerning", color="#f39c12")
    ax.bar(
        x,
        problematic,
        width,
        bottom=np.array(aligned) + np.array(concerning),
        label="Problematic",
        color="#e74c3c",
    )

    # Customize
    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Behavioral Alignment Distribution by Model", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split("/")[-1] for m in models], rotation=0)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 100)

    # Add value labels
    for i, model in enumerate(models):
        total = model_stats[model]["total"]
        if total > 0:
            y_offset: float = 0.0
            for value in (aligned[i], concerning[i], problematic[i]):
                if value > MIN_LABEL_PERCENT:  # Only show label if segment is large enough
                    ax.text(i, y_offset + value / 2, f"{value:.1f}%", ha="center", va="center", fontsize=10)
                y_offset += value

    plt.tight_layout()
    output_file = output_dir / "alignment_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ {output_file.name}")


def plot_category_heatmap(category_stats: dict[str, dict[str, dict[str, int]]], output_dir: Path) -> None:
    """Create heatmap of alignment rates by category and model."""
    categories = sorted(category_stats.keys())
    models = sorted({model for category in category_stats.values() for model in category})

    # Compute alignment percentages
    data = []
    for category in categories:
        row = []
        for model in models:
            stats = category_stats[category][model]
            total = stats["total"]
            if total > 0:
                aligned_pct = (stats["aligned"] / total) * 100
                row.append(aligned_pct)
            else:
                row.append(0)
        data.append(row)

    # Create heatmap
    _fig, ax = plt.subplots(figsize=(10, 7))

    # Use reversed colormap so green = high alignment
    sns.heatmap(
        data,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        xticklabels=[m.split("/")[-1] for m in models],
        yticklabels=[c.replace("_", " ").title() for c in categories],
        vmin=0,
        vmax=100,
        cbar_kws={"label": "Alignment Rate (%)"},
        ax=ax,
    )

    ax.set_title("Behavioral Alignment by Category and Model", fontsize=14)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Behavioral Category", fontsize=12)

    plt.tight_layout()
    output_file = output_dir / "category_heatmap.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ {output_file.name}")


def plot_concern_breakdown(category_stats: dict[str, dict[str, dict[str, int]]], output_dir: Path) -> None:
    """Create stacked bar chart showing concerning/problematic issues by category."""
    categories = sorted(category_stats.keys())

    # Aggregate across all models
    concerning_counts = []
    problematic_counts = []

    for category in categories:
        concerning = 0
        problematic = 0

        for model_stats in category_stats[category].values():
            concerning += model_stats["concerning"]
            problematic += model_stats["problematic"]

        concerning_counts.append(concerning)
        problematic_counts.append(problematic)

    # Create plot
    _fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.6

    ax.bar(x, concerning_counts, width, label="Concerning", color="#f39c12")
    ax.bar(x, problematic_counts, width, bottom=concerning_counts, label="Problematic", color="#e74c3c")

    # Customize
    ax.set_ylabel("Number of Issues", fontsize=12)
    ax.set_title("Behavioral Concerns by Category", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("_", " ").title() for c in categories], rotation=45, ha="right")
    ax.legend(loc="upper right")

    # Add value labels
    for i in range(len(categories)):
        if concerning_counts[i] > 0:
            ax.text(i, concerning_counts[i] / 2, str(concerning_counts[i]), ha="center", va="center", fontsize=10)
        if problematic_counts[i] > 0:
            ax.text(
                i,
                concerning_counts[i] + problematic_counts[i] / 2,
                str(problematic_counts[i]),
                ha="center",
                va="center",
                fontsize=10,
            )

    plt.tight_layout()
    output_file = output_dir / "concern_breakdown.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ {output_file.name}")


def _build_radar_dataframe(results: list[dict[str, Any]]) -> pd.DataFrame:
    data = []
    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown").replace("ollama/", "")
        category = result.get("category", "unknown")
        alignment = result.get("behavioral_alignment")

        # Only include results with valid classifications
        if alignment in ["aligned", "concerning", "problematic"]:
            data.append({"model": model, "category": category, "alignment": alignment})

    return pd.DataFrame(data)


def _calculate_radar_angles(num_categories: int) -> list[float]:
    angles = [float(x) for x in np.linspace(0, 2 * np.pi, num_categories, endpoint=False)]
    angles += angles[:1]
    return angles


def _calculate_alignment_rates(
    df: pd.DataFrame,
    model: str,
    categories: list[str],
) -> tuple[list[float], list[float], list[float]]:
    aligned_rates = []
    concerning_rates = []
    problematic_rates = []

    for category in categories:
        subset = df[(df["model"] == model) & (df["category"] == category)]
        total = len(subset)

        if total > 0:
            aligned = len(subset[subset["alignment"] == "aligned"])
            concerning = len(subset[subset["alignment"] == "concerning"])
            problematic = len(subset[subset["alignment"] == "problematic"])

            aligned_rates.append((aligned / total) * 100)
            concerning_rates.append((concerning / total) * 100)
            problematic_rates.append((problematic / total) * 100)
        else:
            aligned_rates.append(0)
            concerning_rates.append(0)
            problematic_rates.append(0)

    return aligned_rates, concerning_rates, problematic_rates


def _format_radar_stats(df: pd.DataFrame, model: str) -> str:
    total_tests = len(df[df["model"] == model])
    aligned_count = len(df[(df["model"] == model) & (df["alignment"] == "aligned")])
    concerning_count = len(df[(df["model"] == model) & (df["alignment"] == "concerning")])
    problematic_count = len(df[(df["model"] == model) & (df["alignment"] == "problematic")])

    return (
        f"A:{aligned_count}/{total_tests} ({aligned_count/total_tests*100:.1f}%) "
        f"C:{concerning_count} ({concerning_count/total_tests*100:.1f}%) "
        f"P:{problematic_count} ({problematic_count/total_tests*100:.1f}%)"
    )


def plot_radar_charts(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot radar charts with stacked alignment rings showing all alignment types."""
    df = _build_radar_dataframe(results)

    models = sorted(df["model"].unique())
    categories = sorted(df["category"].unique())
    angles = _calculate_radar_angles(len(categories))

    fig, axes = plt.subplots(2, 2, figsize=(14, 14), subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    colors = {"aligned": "#2ecc71", "concerning": "#f39c12", "problematic": "#e74c3c"}

    for idx, model in enumerate(models):
        ax = axes[idx]

        aligned_rates, concerning_rates, problematic_rates = _calculate_alignment_rates(df, model, categories)

        aligned_rates += aligned_rates[:1]
        concerning_rates += concerning_rates[:1]
        problematic_rates += problematic_rates[:1]

        center = [0.0] * len(angles)
        aligned_outer = aligned_rates[:]
        concerning_outer = [a + c for a, c in zip(aligned_rates, concerning_rates, strict=True)]
        problematic_outer = [RADAR_MAX_PERCENT] * len(angles)

        ax.fill_between(angles, center, aligned_outer, alpha=0.7, color=colors["aligned"], label="Aligned")
        ax.fill_between(
            angles,
            aligned_outer,
            concerning_outer,
            alpha=0.7,
            color=colors["concerning"],
            label="Concerning",
        )
        ax.fill_between(
            angles,
            concerning_outer,
            problematic_outer,
            alpha=0.7,
            color=colors["problematic"],
            label="Problematic",
        )

        ax.plot(angles, aligned_outer, "o-", linewidth=1.5, color="#27ae60", alpha=0.8, markersize=4)
        ax.plot(angles, concerning_outer, "o-", linewidth=1.5, color="#e67e22", alpha=0.8, markersize=4)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace("_", " ").title() for c in categories], size=10)
        ax.set_ylim(0, RADAR_MAX_PERCENT)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20%", "40%", "60%", "80%", "100%"], size=9.5)
        ax.grid(True, linestyle="-", alpha=0.5, linewidth=0.7, color="gray")

        ax.text(0.5, 1.17, model, transform=ax.transAxes, ha="center", va="top", fontsize=12, fontweight="bold")
        ax.text(0.5, 1.12, _format_radar_stats(df, model), transform=ax.transAxes, ha="center", va="top", fontsize=9)

    for idx in range(len(models), 4):
        axes[idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        fontsize=11,
        frameon=True,
        title="Alignment Type",
    )

    plt.suptitle(
        "Behavioral Alignment Performance: Category-Specific Alignment Distribution",
        fontsize=14,
        y=0.99,
        fontweight="bold",
    )
    plt.tight_layout(rect=(0, 0, 0.88, 0.97))

    output_file = output_dir / "radar_charts.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"  ✓ {output_file.name}")


def _write_overall_alignment_section(f: TextIO, model_stats: dict[str, dict[str, int]]) -> None:
    total_tests = sum(stats["total"] for stats in model_stats.values())
    total_aligned = sum(stats["aligned"] for stats in model_stats.values())
    total_concerning = sum(stats["concerning"] for stats in model_stats.values())
    total_problematic = sum(stats["problematic"] for stats in model_stats.values())

    f.write("## Overall Alignment Distribution\n\n")
    f.write(f"- Aligned: {total_aligned} ({total_aligned/total_tests*100:.1f}%)\n")
    f.write(f"- Concerning: {total_concerning} ({total_concerning/total_tests*100:.1f}%)\n")
    f.write(f"- Problematic: {total_problematic} ({total_problematic/total_tests*100:.1f}%)\n\n")


def _write_model_performance_section(f: TextIO, model_stats: dict[str, dict[str, int]]) -> None:
    f.write("## Model Performance\n\n")

    for model in sorted(model_stats.keys()):
        stats = model_stats[model]
        total = stats["total"]

        if total == 0:
            continue

        model_name = model.replace("ollama/", "")

        f.write(f"\n### {model_name}\n")
        f.write(f"- Total: {total}\n")
        f.write(f"- Aligned: {stats['aligned']} ({stats['aligned']/total*100:.1f}%)\n")
        f.write(f"- Concerning: {stats['concerning']} ({stats['concerning']/total*100:.1f}%)\n")
        f.write(f"- Problematic: {stats['problematic']} ({stats['problematic']/total*100:.1f}%)\n")


def _write_category_analysis_section(
    f: TextIO,
    category_stats: dict[str, dict[str, dict[str, int]]],
) -> None:
    f.write("\n## Category Analysis\n\n")

    for category in sorted(category_stats.keys()):
        cat_aligned = sum(stats["aligned"] for stats in category_stats[category].values())
        cat_concerning = sum(stats["concerning"] for stats in category_stats[category].values())
        cat_problematic = sum(stats["problematic"] for stats in category_stats[category].values())
        cat_total = sum(stats["total"] for stats in category_stats[category].values())

        if cat_total == 0:
            continue

        category_name = category.replace("_", " ").title()
        alignment_rate = cat_aligned / cat_total * 100

        f.write(f"\n### {category_name}\n")
        f.write(f"- Overall alignment rate: {alignment_rate:.1f}%\n")
        f.write(f"- Concerning: {cat_concerning}/{cat_total}\n")
        f.write(f"- Problematic: {cat_problematic}/{cat_total}\n\n")

        worst_model = None
        worst_rate = 100.0

        for model, stats in category_stats[category].items():
            if stats["total"] > 0:
                aligned_rate = stats["aligned"] / stats["total"] * 100
                if aligned_rate < worst_rate:
                    worst_rate = aligned_rate
                    worst_model = model.replace("ollama/", "")

        if worst_model:
            f.write(f"- Most vulnerable model: {worst_model} ({worst_rate:.1f}% aligned)\n")


def generate_statistics_report(
    model_stats: dict[str, dict[str, int]],
    category_stats: dict[str, dict[str, dict[str, int]]],
    output_dir: Path,
) -> None:
    """Generate statistics report in markdown format."""
    report_file = output_dir / "statistics_report.md"

    total_tests = sum(stats["total"] for stats in model_stats.values())
    num_models = len([m for m in model_stats.values() if m["total"] > 0])
    num_categories = len(category_stats)

    with report_file.open("w") as f:
        f.write("# Behavioral Safety Evaluation Analysis Report\n\n")
        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Models tested: {num_models}\n")
        f.write(f"Behavioral categories: {num_categories}\n\n")

        _write_overall_alignment_section(f, model_stats)
        _write_model_performance_section(f, model_stats)
        _write_category_analysis_section(f, category_stats)

    logger.info(f"  ✓ {report_file.name}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyse behavioral evaluation results and generate visualizations")
    parser.add_argument(
        "results_file",
        nargs="?",
        type=Path,
        help="Path to results JSON file (default: most recent in results/raw/)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Load configuration
    config = load_config()

    # Load results
    logger.info("Loading results...")
    results = load_results(args.results_file)
    logger.info(f"Loaded {len(results)} test results")

    # Compute statistics
    logger.info("Computing statistics...")
    model_stats = compute_alignment_stats(results)
    category_stats = compute_category_stats(results)
    _flag_stats = compute_flag_stats(results)

    # Ensure output directories exist
    figures_dir = SCRIPT_DIR / config["paths"]["figures"]
    analysis_dir = SCRIPT_DIR / config["paths"]["analysis"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating visualizations...")

    # Generate plots (save to figures/)
    plot_alignment_distribution(model_stats, figures_dir)
    plot_category_heatmap(category_stats, figures_dir)
    plot_concern_breakdown(category_stats, figures_dir)
    plot_radar_charts(results, figures_dir)

    # Generate statistics report (save to analysis/)
    generate_statistics_report(model_stats, category_stats, analysis_dir)

    logger.info("=" * 60)
    logger.info("Analysis complete!")
    logger.info("=" * 60)
    logger.info(f"Figures saved to: {figures_dir}/")
    logger.info(f"Analysis saved to: {analysis_dir}/")
    logger.info("\nGenerated files:")
    logger.info("  Figures:")
    logger.info("    - alignment_distribution.png")
    logger.info("    - category_heatmap.png")
    logger.info("    - concern_breakdown.png")
    logger.info("    - radar_charts.png")
    logger.info("  Analysis:")
    logger.info("    - statistics_report.md")


if __name__ == "__main__":
    main()
