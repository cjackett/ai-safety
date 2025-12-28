"""
Analysis and visualization for behavioral safety evaluation results.

Generates:
- Alignment distribution charts by model
- Category-specific heatmaps
- Behavioral flag analysis
- Detailed findings per category
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Script directory
SCRIPT_DIR = Path(__file__).parent


def load_results(results_file: Path = None) -> list[dict[str, Any]]:
    """Load behavioral evaluation results from JSON file."""
    if results_file is None:
        # Find most recent results file
        results_dir = SCRIPT_DIR / "results"
        json_files = list(results_dir.glob("behavioral_eval*.json"))

        if not json_files:
            raise FileNotFoundError(
                f"No results files found in {results_dir}. "
                "Run behavioral_eval.py first to generate results."
            )

        results_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected results file: {results_file.name}")

    with open(results_file, "r") as f:
        return json.load(f)


def compute_alignment_stats(results: list[dict[str, Any]]) -> dict:
    """Compute overall alignment statistics by model."""
    model_stats = defaultdict(lambda: {
        "total": 0,
        "aligned": 0,
        "concerning": 0,
        "problematic": 0,
    })

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        alignment = result.get("behavioral_alignment")

        model_stats[model]["total"] += 1

        if alignment == "aligned":
            model_stats[model]["aligned"] += 1
        elif alignment == "concerning":
            model_stats[model]["concerning"] += 1
        elif alignment == "problematic":
            model_stats[model]["problematic"] += 1

    return model_stats


def compute_category_stats(results: list[dict[str, Any]]) -> dict:
    """Compute alignment statistics by category and model."""
    category_stats = defaultdict(lambda: defaultdict(lambda: {
        "aligned": 0,
        "concerning": 0,
        "problematic": 0,
        "total": 0,
    }))

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        category = result.get("category", "unknown")
        alignment = result.get("behavioral_alignment")

        category_stats[category][model]["total"] += 1

        if alignment == "aligned":
            category_stats[category][model]["aligned"] += 1
        elif alignment == "concerning":
            category_stats[category][model]["concerning"] += 1
        elif alignment == "problematic":
            category_stats[category][model]["problematic"] += 1

    return category_stats


def compute_flag_stats(results: list[dict[str, Any]]) -> dict:
    """Compute behavioral flag statistics."""
    flag_stats = defaultdict(lambda: defaultdict(int))

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        flags = result.get("behavioral_flags", [])

        for flag in flags:
            flag_stats[model][flag] += 1

    return flag_stats


def plot_alignment_distribution(
    model_stats: dict,
    output_dir: Path
) -> None:
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
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.6

    # Stack bars
    p1 = ax.bar(x, aligned, width, label='Aligned', color='#2ecc71')
    p2 = ax.bar(x, concerning, width, bottom=aligned, label='Concerning', color='#f39c12')
    p3 = ax.bar(x, problematic, width, bottom=np.array(aligned) + np.array(concerning),
                label='Problematic', color='#e74c3c')

    # Customize
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Behavioral Alignment Distribution by Model', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split('/')[-1] for m in models], rotation=0)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add value labels
    for i, model in enumerate(models):
        total = model_stats[model]["total"]
        if total > 0:
            y_offset = 0
            for value, color in [(aligned[i], '#2ecc71'),
                                 (concerning[i], '#f39c12'),
                                 (problematic[i], '#e74c3c')]:
                if value > 3:  # Only show label if segment is large enough
                    ax.text(i, y_offset + value/2, f'{value:.1f}%',
                           ha='center', va='center', fontsize=10)
                y_offset += value

    plt.tight_layout()
    plt.savefig(output_dir / "alignment_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ alignment_distribution.png")


def plot_category_heatmap(
    category_stats: dict,
    output_dir: Path
) -> None:
    """Create heatmap of alignment rates by category and model."""
    categories = sorted(category_stats.keys())
    models = sorted(set(
        model
        for category in category_stats.values()
        for model in category.keys()
    ))

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
    fig, ax = plt.subplots(figsize=(10, 7))

    # Use reversed colormap so green = high alignment
    sns.heatmap(
        data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn',
        xticklabels=[m.split('/')[-1] for m in models],
        yticklabels=[c.replace('_', ' ').title() for c in categories],
        vmin=0,
        vmax=100,
        cbar_kws={'label': 'Alignment Rate (%)'},
        ax=ax
    )

    ax.set_title('Behavioral Alignment by Category and Model', fontsize=14)
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Behavioral Category', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "category_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ category_heatmap.png")


def plot_concern_breakdown(
    category_stats: dict,
    output_dir: Path
) -> None:
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
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.6

    p1 = ax.bar(x, concerning_counts, width, label='Concerning', color='#f39c12')
    p2 = ax.bar(x, problematic_counts, width, bottom=concerning_counts,
                label='Problematic', color='#e74c3c')

    # Customize
    ax.set_ylabel('Number of Issues', fontsize=12)
    ax.set_title('Behavioral Concerns by Category', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], rotation=45, ha='right')
    ax.legend(loc='upper right')

    # Add value labels
    for i in range(len(categories)):
        if concerning_counts[i] > 0:
            ax.text(i, concerning_counts[i]/2, str(concerning_counts[i]),
                   ha='center', va='center', fontsize=10)
        if problematic_counts[i] > 0:
            ax.text(i, concerning_counts[i] + problematic_counts[i]/2,
                   str(problematic_counts[i]),
                   ha='center', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "concern_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ concern_breakdown.png")


def plot_radar_charts(
    results: list[dict[str, Any]],
    output_dir: Path
) -> None:
    """Plot radar charts with stacked alignment rings showing all alignment types."""
    # Extract data from results
    import pandas as pd

    data = []
    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown").replace("ollama/", "")
        category = result.get("category", "unknown")
        alignment = result.get("behavioral_alignment")

        data.append({
            "model": model,
            "category": category,
            "alignment": alignment
        })

    df = pd.DataFrame(data)

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

    # Colors for each alignment type
    colors = {
        "aligned": "#2ecc71",      # Green
        "concerning": "#f39c12",   # Orange
        "problematic": "#e74c3c"   # Red
    }

    for idx, model in enumerate(models):
        ax = axes[idx]

        # Calculate alignment rates for each category
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

                # Calculate percentages
                aligned_rates.append((aligned / total) * 100)
                concerning_rates.append((concerning / total) * 100)
                problematic_rates.append((problematic / total) * 100)
            else:
                aligned_rates.append(0)
                concerning_rates.append(0)
                problematic_rates.append(0)

        # Close the polygons
        aligned_rates += aligned_rates[:1]
        concerning_rates += concerning_rates[:1]
        problematic_rates += problematic_rates[:1]

        # Create stacked rings from center outward:
        # 1. Inner ring (0% to aligned%): aligned area (green) - safe at center
        # 2. Middle ring (aligned% to aligned%+concerning%): concerning area (orange)
        # 3. Outer ring (aligned%+concerning% to 100%): problematic (red) - dangerous at edge

        # Calculate cumulative percentages for stacking
        center = [0] * len(angles)
        aligned_outer = aligned_rates[:]
        concerning_outer = [(a + c) for a, c in zip(aligned_rates, concerning_rates)]
        problematic_outer = [100] * len(angles)

        # Plot stacked areas from inside to outside
        ax.fill_between(angles, center, aligned_outer,
                        alpha=0.7, color=colors["aligned"], label='Aligned')
        ax.fill_between(angles, aligned_outer, concerning_outer,
                        alpha=0.7, color=colors["concerning"], label='Concerning')
        ax.fill_between(angles, concerning_outer, problematic_outer,
                        alpha=0.7, color=colors["problematic"], label='Problematic')

        # Add edge lines for clarity
        ax.plot(angles, aligned_outer, 'o-', linewidth=1.5, color='#27ae60', alpha=0.8, markersize=4)
        ax.plot(angles, concerning_outer, 'o-', linewidth=1.5, color='#e67e22', alpha=0.8, markersize=4)

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], size=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=9.5)
        ax.grid(True, linestyle='-', alpha=0.5, linewidth=0.7, color='gray')

        # Title with alignment summary
        total_tests = len(df[df["model"] == model])
        aligned_count = len(df[(df["model"] == model) & (df["alignment"] == "aligned")])
        concerning_count = len(df[(df["model"] == model) & (df["alignment"] == "concerning")])
        problematic_count = len(df[(df["model"] == model) & (df["alignment"] == "problematic")])

        # Set model name as bold text at top
        ax.text(0.5, 1.17, model, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, fontweight='bold')

        # Add stats below the model name
        stats_text = f"A:{aligned_count}/{total_tests} ({aligned_count/total_tests*100:.1f}%) "
        stats_text += f"C:{concerning_count} ({concerning_count/total_tests*100:.1f}%) "
        stats_text += f"P:{problematic_count} ({problematic_count/total_tests*100:.1f}%)"
        ax.text(0.5, 1.12, stats_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=9)

    # Hide unused subplots if fewer than 4 models
    for idx in range(num_models, 4):
        axes[idx].set_visible(False)

    # Add legend to the figure (top right, outside all plots)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
              fontsize=11, frameon=True, title='Alignment Type')

    plt.suptitle("Behavioral Alignment Performance: Category-Specific Alignment Distribution",
                 fontsize=14, y=0.99, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    plt.savefig(output_dir / "radar_charts.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  ✓ radar_charts.png")


def plot_flag_analysis(
    flag_stats: dict,
    output_dir: Path
) -> None:
    """Create horizontal bar chart of top behavioral flags."""
    # Aggregate flags across all models
    all_flags = defaultdict(int)

    for model_flags in flag_stats.values():
        for flag, count in model_flags.items():
            all_flags[flag] += count

    # Get top 10 flags
    top_flags = sorted(all_flags.items(), key=lambda x: x[1], reverse=True)[:10]

    if not top_flags:
        print("  ⚠ No behavioral flags to visualize")
        return

    flags, counts = zip(*top_flags)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    y_pos = np.arange(len(flags))

    # Color code based on flag type
    colors = []
    for flag in flags:
        if any(word in flag for word in ['problematic', 'fabricate', 'deception', 'power', 'resist']):
            colors.append('#e74c3c')  # Red for problematic
        elif any(word in flag for word in ['concerning', 'overconfident', 'spurious']):
            colors.append('#f39c12')  # Orange for concerning
        else:
            colors.append('#3498db')  # Blue for neutral/aligned

    ax.barh(y_pos, counts, color=colors)

    # Customize
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ').title() for f in flags])
    ax.invert_yaxis()
    ax.set_xlabel('Occurrences', fontsize=12)
    ax.set_title('Top Behavioral Flags Detected', fontsize=14)

    # Add value labels
    for i, count in enumerate(counts):
        ax.text(count + 0.5, i, str(count), va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "flag_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ flag_analysis.png")


def generate_statistics_report(
    results: list[dict[str, Any]],
    model_stats: dict,
    category_stats: dict,
    output_dir: Path
) -> None:
    """Generate statistics report in markdown format."""
    report_file = output_dir / "statistics_report.md"

    # Count total tests
    total_tests = sum(stats["total"] for stats in model_stats.values())
    num_models = len([m for m in model_stats.values() if m["total"] > 0])
    num_categories = len(category_stats)

    with open(report_file, "w") as f:
        f.write("# Behavioral Safety Evaluation Analysis Report\n\n")

        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Models tested: {num_models}\n")
        f.write(f"Behavioral categories: {num_categories}\n\n")

        # Overall alignment distribution
        f.write("## Overall Alignment Distribution\n\n")

        total_aligned = sum(stats["aligned"] for stats in model_stats.values())
        total_concerning = sum(stats["concerning"] for stats in model_stats.values())
        total_problematic = sum(stats["problematic"] for stats in model_stats.values())

        f.write(f"- aligned: {total_aligned} ({total_aligned/total_tests*100:.1f}%)\n")
        f.write(f"- concerning: {total_concerning} ({total_concerning/total_tests*100:.1f}%)\n")
        f.write(f"- problematic: {total_problematic} ({total_problematic/total_tests*100:.1f}%)\n\n")

        # Model performance
        f.write("## Model Performance\n\n")

        for model in sorted(model_stats.keys()):
            stats = model_stats[model]
            total = stats["total"]

            if total == 0:
                continue

            # Clean up model name
            model_name = model.replace("ollama/", "")

            f.write(f"\n### {model_name}\n")
            f.write(f"- Total: {total}\n")
            f.write(f"- Aligned: {stats['aligned']} ({stats['aligned']/total*100:.1f}%)\n")
            f.write(f"- Concerning: {stats['concerning']} ({stats['concerning']/total*100:.1f}%)\n")
            f.write(f"- Problematic: {stats['problematic']} ({stats['problematic']/total*100:.1f}%)\n")

        # Category analysis
        f.write("\n## Category Analysis\n\n")

        for category in sorted(category_stats.keys()):
            # Calculate overall category stats
            cat_aligned = sum(stats["aligned"] for stats in category_stats[category].values())
            cat_concerning = sum(stats["concerning"] for stats in category_stats[category].values())
            cat_problematic = sum(stats["problematic"] for stats in category_stats[category].values())
            cat_total = sum(stats["total"] for stats in category_stats[category].values())

            if cat_total == 0:
                continue

            category_name = category.replace('_', ' ').title()
            alignment_rate = cat_aligned / cat_total * 100

            f.write(f"\n### {category_name}\n")
            f.write(f"- Overall alignment rate: {alignment_rate:.1f}%\n")
            f.write(f"- Concerning: {cat_concerning}/{cat_total}\n")
            f.write(f"- Problematic: {cat_problematic}/{cat_total}\n\n")

            # Find most vulnerable model in this category
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

    print(f"  ✓ statistics_report.md")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse behavioral evaluation results and generate visualizations"
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        type=Path,
        help="Path to results JSON file (default: most recent in results/)"
    )

    args = parser.parse_args()

    # Load results
    results = load_results(args.results_file)

    print(f"\nLoaded {len(results)} test results")
    print(f"\nComputing statistics...")

    # Compute statistics
    model_stats = compute_alignment_stats(results)
    category_stats = compute_category_stats(results)
    flag_stats = compute_flag_stats(results)

    # Ensure output directory exists
    output_dir = SCRIPT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Generate plots
    plot_alignment_distribution(model_stats, output_dir)
    plot_category_heatmap(category_stats, output_dir)
    plot_concern_breakdown(category_stats, output_dir)
    plot_flag_analysis(flag_stats, output_dir)
    plot_radar_charts(results, output_dir)

    # Generate statistics report
    generate_statistics_report(results, model_stats, category_stats, output_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")
    print(f"Results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - alignment_distribution.png")
    print("  - category_heatmap.png")
    print("  - concern_breakdown.png")
    print("  - flag_analysis.png")
    print("  - statistics_report.md")


if __name__ == "__main__":
    main()
