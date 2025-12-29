"""
Analysis and visualization for multimodal safety evaluation results.

Generates:
- Compliance distribution charts by model
- Attack type effectiveness comparison
- Radar charts showing category-specific performance
- Comparison with text-only models from Experiment 01
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Script directory
SCRIPT_DIR = Path(__file__).parent


def load_results(results_file: Path = None) -> list[dict[str, Any]]:
    """Load multimodal evaluation results from JSON file."""
    if results_file is None:
        # Find most recent results file
        results_dir = SCRIPT_DIR / "results"
        json_files = list(results_dir.glob("vision_eval*.json"))

        if not json_files:
            raise FileNotFoundError(
                f"No results files found in {results_dir}. "
                "Run vision_jailbreaks.py first to generate results."
            )

        results_file = max(json_files, key=lambda p: p.stat().st_mtime)
        print(f"Auto-detected results file: {results_file.name}")

    with open(results_file, "r") as f:
        return json.load(f)


def compute_compliance_stats(results: list[dict[str, Any]]) -> dict:
    """Compute compliance statistics by model."""
    model_stats = defaultdict(lambda: {
        "total": 0,
        "refused": 0,
        "hedged": 0,
        "full": 0,
    })

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        compliance = result.get("compliance_type")

        model_stats[model]["total"] += 1

        if compliance == "refused":
            model_stats[model]["refused"] += 1
        elif compliance == "hedged":
            model_stats[model]["hedged"] += 1
        elif compliance == "full":
            model_stats[model]["full"] += 1

    return model_stats


def compute_attack_type_stats(results: list[dict[str, Any]]) -> dict:
    """Compute success rates by attack type."""
    attack_stats = defaultdict(lambda: {
        "total": 0,
        "refused": 0,
        "hedged": 0,
        "full": 0,
    })

    for result in results:
        if result.get("error"):
            continue

        attack_type = result.get("attack_type", "unknown")
        compliance = result.get("compliance_type")

        attack_stats[attack_type]["total"] += 1

        if compliance == "refused":
            attack_stats[attack_type]["refused"] += 1
        elif compliance == "hedged":
            attack_stats[attack_type]["hedged"] += 1
        elif compliance == "full":
            attack_stats[attack_type]["full"] += 1

    return attack_stats


def compute_category_stats(results: list[dict[str, Any]]) -> dict:
    """Compute compliance statistics by category and model."""
    category_stats = defaultdict(lambda: defaultdict(lambda: {
        "refused": 0,
        "hedged": 0,
        "full": 0,
        "total": 0,
    }))

    for result in results:
        if result.get("error"):
            continue

        model = result.get("model", "unknown")
        category = result.get("category", "unknown")
        compliance = result.get("compliance_type")

        category_stats[category][model]["total"] += 1

        if compliance == "refused":
            category_stats[category][model]["refused"] += 1
        elif compliance == "hedged":
            category_stats[category][model]["hedged"] += 1
        elif compliance == "full":
            category_stats[category][model]["full"] += 1

    return category_stats


def plot_compliance_distribution(model_stats: dict, output_dir: Path) -> None:
    """Create stacked bar chart of compliance distribution by model."""
    models = sorted(model_stats.keys())
    refused = []
    hedged = []
    full = []

    for model in models:
        stats = model_stats[model]
        total = stats["total"]

        if total > 0:
            refused.append((stats["refused"] / total) * 100)
            hedged.append((stats["hedged"] / total) * 100)
            full.append((stats["full"] / total) * 100)
        else:
            refused.append(0)
            hedged.append(0)
            full.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(models))
    width = 0.6

    # Stack bars
    p1 = ax.bar(x, refused, width, label='Refused', color='#2ecc71')
    p2 = ax.bar(x, hedged, width, bottom=refused, label='Hedged', color='#f39c12')
    p3 = ax.bar(x, full, width, bottom=np.array(refused) + np.array(hedged),
                label='Full Compliance', color='#e74c3c')

    # Customize
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Multimodal Safety: Compliance Distribution by Model\n(Excluding benign control tests)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.split(':')[0] for m in models], rotation=0)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add value labels
    for i, model in enumerate(models):
        total = model_stats[model]["total"]
        if total > 0:
            y_offset = 0
            for value, color in [(refused[i], '#2ecc71'),
                                 (hedged[i], '#f39c12'),
                                 (full[i], '#e74c3c')]:
                if value > 3:  # Only show label if segment is large enough
                    ax.text(i, y_offset + value/2, f'{value:.1f}%',
                           ha='center', va='center', fontsize=10)
                y_offset += value

    plt.tight_layout()
    plt.savefig(output_dir / "compliance_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ compliance_distribution.png")


def plot_attack_effectiveness(attack_stats: dict, output_dir: Path) -> None:
    """Create bar chart showing attack type success rates."""
    attack_types = sorted(attack_stats.keys())
    success_rates = []

    for attack_type in attack_types:
        stats = attack_stats[attack_type]
        total = stats["total"]
        if total > 0:
            # Success = hedged + full compliance
            success = (stats["hedged"] + stats["full"]) / total * 100
            success_rates.append(success)
        else:
            success_rates.append(0)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(attack_types))
    bars = ax.bar(x, success_rates, color='#e74c3c', alpha=0.8)

    # Customize
    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title('Multimodal Attack Effectiveness by Type\n(Excluding benign control tests)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace('_', ' ').title() for a in attack_types],
                        rotation=45, ha='right')
    ax.set_ylim(0, max(success_rates) + 10 if success_rates else 10)

    # Add value labels
    for bar, rate in zip(bars, success_rates):
        if rate > 0:
            ax.text(bar.get_x() + bar.get_width()/2., rate + 1,
                   f'{rate:.1f}%',
                   ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / "attack_effectiveness.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("  ✓ attack_effectiveness.png")


def plot_radar_charts(results: list[dict[str, Any]], output_dir: Path) -> None:
    """Plot radar charts with stacked compliance rings showing all compliance types.

    Note: results should already have benign control tests filtered out.
    """
    # Extract unique models and categories
    models = sorted(set(r["model"] for r in results if not r.get("error")))
    categories = sorted(set(r["category"] for r in results if not r.get("error")))

    num_categories = len(categories)
    num_models = len(models)

    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create subplots in a single row for 3 models
    if num_models <= 3:
        fig, axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6),
                                 subplot_kw=dict(projection='polar'))
        if num_models == 1:
            axes = [axes]  # Make it iterable
    else:
        # Fall back to 2x2 grid for 4+ models
        fig, axes = plt.subplots(2, 2, figsize=(14, 14),
                                 subplot_kw=dict(projection='polar'))
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
            subset = [r for r in results
                     if r["model"] == model and r["category"] == category and not r.get("error")]
            total = len(subset)

            if total > 0:
                refused = len([r for r in subset if r["compliance_type"] == "refused"])
                hedged = len([r for r in subset if r["compliance_type"] == "hedged"])
                full = len([r for r in subset if r["compliance_type"] == "full"])

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

        # Create stacked rings from center outward
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

        # Title with compliance summary (benign control already filtered out)
        total_tests = len([r for r in results if r["model"] == model and not r.get("error")])
        refused_count = len([r for r in results if r["model"] == model and r["compliance_type"] == "refused"])
        hedged_count = len([r for r in results if r["model"] == model and r["compliance_type"] == "hedged"])
        full_count = len([r for r in results if r["model"] == model and r["compliance_type"] == "full"])

        # Set model name as bold text at top
        model_name = model.split(':')[0]  # Remove version tag
        ax.text(0.5, 1.17, model_name, transform=ax.transAxes,
                ha='center', va='top', fontsize=12, fontweight='bold')

        # Add stats below the model name
        if total_tests > 0:
            stats_text = f"R:{refused_count}/{total_tests} ({refused_count/total_tests*100:.1f}%) "
            stats_text += f"H:{hedged_count} ({hedged_count/total_tests*100:.1f}%) "
            stats_text += f"F:{full_count} ({full_count/total_tests*100:.1f}%)"
            ax.text(0.5, 1.12, stats_text, transform=ax.transAxes,
                    ha='center', va='top', fontsize=9)

    # Hide unused subplots only for 2x2 grid layout (4+ models)
    if num_models > 3:
        for idx in range(num_models, 4):
            axes[idx].set_visible(False)

    # Add legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    if num_models <= 3:
        # Single row layout - legend at top center
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02),
                  fontsize=11, frameon=True, title='Compliance Type', ncol=3)
        plt.suptitle("Multimodal Safety Performance: Category-Specific Compliance Distribution\n(Excluding benign control tests)",
                     fontsize=14, y=0.98, fontweight='bold')
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    else:
        # 2x2 grid layout - legend at top right
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.98),
                  fontsize=11, frameon=True, title='Compliance Type')
        plt.suptitle("Multimodal Safety Performance: Category-Specific Compliance Distribution\n(Excluding benign control tests)",
                     fontsize=14, y=0.99, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 0.88, 0.97])
    plt.savefig(output_dir / "radar_charts.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("  ✓ radar_charts.png")


def generate_statistics_report(
    results: list[dict[str, Any]],
    model_stats: dict,
    attack_stats: dict,
    category_stats: dict,
    output_dir: Path
) -> None:
    """Generate statistics report in markdown format."""
    report_file = output_dir / "statistics_report.md"

    # Count total tests
    total_tests = sum(stats["total"] for stats in model_stats.values())
    num_models = len([m for m in model_stats.values() if m["total"] > 0])
    num_attack_types = len(attack_stats)

    with open(report_file, "w") as f:
        f.write("# Multimodal Safety Evaluation Analysis Report\n\n")

        f.write(f"Total tests: {total_tests}\n")
        f.write(f"Models tested: {num_models}\n")
        f.write(f"Attack types: {num_attack_types}\n")
        f.write(f"Harm categories: {len(category_stats)}\n\n")

        # Overall compliance distribution
        f.write("## Overall Compliance Distribution\n\n")

        total_refused = sum(stats["refused"] for stats in model_stats.values())
        total_hedged = sum(stats["hedged"] for stats in model_stats.values())
        total_full = sum(stats["full"] for stats in model_stats.values())

        f.write(f"- refused: {total_refused} ({total_refused/total_tests*100:.1f}%)\n")
        f.write(f"- hedged: {total_hedged} ({total_hedged/total_tests*100:.1f}%)\n")
        f.write(f"- full: {total_full} ({total_full/total_tests*100:.1f}%)\n\n")

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
            f.write(f"- Refused: {stats['refused']} ({stats['refused']/total*100:.1f}%)\n")
            f.write(f"- Hedged: {stats['hedged']} ({stats['hedged']/total*100:.1f}%)\n")
            f.write(f"- Full: {stats['full']} ({stats['full']/total*100:.1f}%)\n")

        # Attack type effectiveness
        f.write("\n## Attack Type Effectiveness\n\n")

        for attack_type in sorted(attack_stats.keys()):
            stats = attack_stats[attack_type]
            total = stats["total"]

            if total == 0:
                continue

            attack_name = attack_type.replace('_', ' ').title()
            success_rate = (stats["hedged"] + stats["full"]) / total * 100

            f.write(f"\n### {attack_name}\n")
            f.write(f"- Total tests: {total}\n")
            f.write(f"- Success rate: {success_rate:.1f}% ({stats['hedged']} hedged + {stats['full']} full)\n")
            f.write(f"- Refusal rate: {stats['refused']/total*100:.1f}%\n")

        # Category analysis
        f.write("\n## Category Analysis\n\n")

        for category in sorted(category_stats.keys()):
            # Calculate overall category stats
            cat_refused = sum(stats["refused"] for stats in category_stats[category].values())
            cat_hedged = sum(stats["hedged"] for stats in category_stats[category].values())
            cat_full = sum(stats["full"] for stats in category_stats[category].values())
            cat_total = sum(stats["total"] for stats in category_stats[category].values())

            if cat_total == 0:
                continue

            category_name = category.replace('_', ' ').title()
            refusal_rate = cat_refused / cat_total * 100

            f.write(f"\n### {category_name}\n")
            f.write(f"- Overall refusal rate: {refusal_rate:.1f}%\n")
            f.write(f"- Hedged: {cat_hedged}/{cat_total}\n")
            f.write(f"- Full: {cat_full}/{cat_total}\n\n")

            # Find most vulnerable model in this category
            worst_model = None
            worst_rate = 100.0

            for model, stats in category_stats[category].items():
                if stats["total"] > 0:
                    refusal_rate = stats["refused"] / stats["total"] * 100
                    if refusal_rate < worst_rate:
                        worst_rate = refusal_rate
                        worst_model = model.split(':')[0]

            if worst_model:
                f.write(f"- Most vulnerable model: {worst_model} ({worst_rate:.1f}% refusal)\n")

        # Multimodal-specific metrics
        f.write("\n## Multimodal-Specific Metrics\n\n")

        # OCR Verification rates
        f.write("### OCR Verification\n\n")

        total_tests_with_ocr = len([r for r in results if r.get('ocr_verified') is not None and not r.get('error')])
        ocr_verified = len([r for r in results if r.get('ocr_verified') == True])
        ocr_failed = len([r for r in results if r.get('ocr_verified') == False])

        if total_tests_with_ocr > 0:
            f.write(f"- OCR verified: {ocr_verified}/{total_tests_with_ocr} ({ocr_verified/total_tests_with_ocr*100:.1f}%)\n")
            f.write(f"- OCR failed: {ocr_failed}/{total_tests_with_ocr} ({ocr_failed/total_tests_with_ocr*100:.1f}%)\n\n")

            # OCR failure by attack type
            attack_ocr_stats = defaultdict(lambda: {"verified": 0, "failed": 0})
            for r in results:
                if r.get('error') or r.get('ocr_verified') is None:
                    continue
                attack_type = r.get('attack_type', 'unknown')
                if r['ocr_verified']:
                    attack_ocr_stats[attack_type]['verified'] += 1
                else:
                    attack_ocr_stats[attack_type]['failed'] += 1

            ocr_failed_types = [(at, stats['failed']) for at, stats in attack_ocr_stats.items() if stats['failed'] > 0]
            if ocr_failed_types:
                f.write("**OCR failures by attack type:**\n")
                for attack_type, count in sorted(ocr_failed_types, key=lambda x: x[1], reverse=True):
                    total = attack_ocr_stats[attack_type]['verified'] + attack_ocr_stats[attack_type]['failed']
                    attack_name = attack_type.replace('_', ' ').title()
                    f.write(f"- {attack_name}: {count}/{total} failures ({count/total*100:.1f}%)\n")
                f.write("\n")

            # OCR failure by model
            model_ocr_stats = defaultdict(lambda: {"verified": 0, "failed": 0})
            for r in results:
                if r.get('error') or r.get('ocr_verified') is None:
                    continue
                model = r.get('model', 'unknown')
                if r['ocr_verified']:
                    model_ocr_stats[model]['verified'] += 1
                else:
                    model_ocr_stats[model]['failed'] += 1

            f.write("**OCR accuracy by model:**\n")
            for model in sorted(model_ocr_stats.keys()):
                stats = model_ocr_stats[model]
                total = stats['verified'] + stats['failed']
                if total > 0:
                    model_name = model.replace("ollama/", "")
                    f.write(f"- {model_name}: {stats['verified']}/{total} verified ({stats['verified']/total*100:.1f}%)\n")
            f.write("\n")

        # Repetition detection
        f.write("### Repetition Detection\n\n")

        repetitive_responses = [r for r in results if r.get('response_repetitive') == True]
        total_tests_checked = len([r for r in results if 'response_repetitive' in r and not r.get('error')])

        if total_tests_checked > 0:
            f.write(f"- Repetitive responses detected: {len(repetitive_responses)}/{total_tests_checked} ({len(repetitive_responses)/total_tests_checked*100:.1f}%)\n\n")

            if repetitive_responses:
                f.write("**Repetitive responses by model:**\n")
                model_rep_count = defaultdict(int)
                for r in repetitive_responses:
                    model_rep_count[r.get('model', 'unknown')] += 1

                for model in sorted(model_rep_count.keys()):
                    model_name = model.replace("ollama/", "")
                    f.write(f"- {model_name}: {model_rep_count[model]} occurrences\n")
                f.write("\n")

                f.write("**Repetitive responses by attack type:**\n")
                attack_rep_count = defaultdict(int)
                for r in repetitive_responses:
                    attack_rep_count[r.get('attack_type', 'unknown')] += 1

                for attack_type in sorted(attack_rep_count.keys()):
                    attack_name = attack_type.replace('_', ' ').title()
                    f.write(f"- {attack_name}: {attack_rep_count[attack_type]} occurrences\n")
                f.write("\n")

    print("  ✓ statistics_report.md")


def main():
    parser = argparse.ArgumentParser(
        description="Analyse multimodal evaluation results and generate visualizations"
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

    # Filter out benign control for safety-focused analysis
    # (benign control tests expect full compliance, which is correct behavior)
    safety_results = [r for r in results if r.get("attack_type") != "benign_control"]
    benign_count = len(results) - len(safety_results)

    print(f"  - Safety tests (harmful): {len(safety_results)}")
    print(f"  - Benign control tests: {benign_count}")
    print(f"\nComputing statistics...")

    # Compute statistics (excluding benign control for safety metrics)
    model_stats = compute_compliance_stats(safety_results)
    attack_stats = compute_attack_type_stats(safety_results)
    category_stats = compute_category_stats(safety_results)

    # Ensure output directory exists
    output_dir = SCRIPT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    print(f"\nGenerating visualizations...")

    # Generate plots (using filtered safety results)
    plot_compliance_distribution(model_stats, output_dir)
    plot_attack_effectiveness(attack_stats, output_dir)
    plot_radar_charts(safety_results, output_dir)

    # Generate statistics report (using all results for benign control reporting)
    generate_statistics_report(results, model_stats, attack_stats, category_stats, output_dir)

    print(f"\n{'='*60}")
    print("Analysis complete!")
    print(f"{'='*60}\n")
    print(f"Results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - compliance_distribution.png")
    print("  - attack_effectiveness.png")
    print("  - radar_charts.png")
    print("  - statistics_report.md")


if __name__ == "__main__":
    main()
