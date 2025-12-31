"""
Analysis and visualization for guardrail testing results.

Generates:
- Effectiveness comparison charts
- False positive analysis
- Latency overhead visualizations
- Configuration comparison
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_summary(summary_file: str) -> Dict:
    """Load summary JSON file."""
    with open(summary_file, 'r') as f:
        return json.load(f)


def plot_jailbreak_effectiveness(summaries: Dict[str, Dict], output_dir: Path):
    """Plot jailbreak resistance by configuration."""
    # Order: permissive, balanced, strict
    config_order = ['permissive', 'balanced', 'strict']
    configs = [c for c in config_order if c in summaries]

    # Extract metrics
    input_blocks = []
    output_blocks = []
    bypasses = []

    for config in configs:
        if "jailbreak_resistance" in summaries[config]:
            jr = summaries[config]["jailbreak_resistance"]
            input_blocks.append(jr["block_rate"] * 100)
            output_blocks.append(jr["output_catch_rate"] * 100)
            bypasses.append(jr["bypass_rate"] * 100)
        else:
            input_blocks.append(0)
            output_blocks.append(0)
            bypasses.append(0)

    # Create stacked bar chart
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.6

    p1 = ax.bar(x, input_blocks, width, label='Blocked at Input', color='#2ecc71')
    p2 = ax.bar(x, output_blocks, width, bottom=input_blocks, label='Blocked at Output', color='#f39c12')
    p3 = ax.bar(x, bypasses, width, bottom=np.array(input_blocks)+np.array(output_blocks),
                label='Bypassed', color='#e74c3c')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Jailbreak Defense Effectiveness by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in configs])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)

    # Add value labels for each layer
    for i, config in enumerate(configs):
        # Label for input blocks (if > 3% to avoid clutter)
        if input_blocks[i] > 3:
            ax.text(i, input_blocks[i]/2, f'{input_blocks[i]:.1f}%',
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)

        # Label for output blocks (if > 3%)
        if output_blocks[i] > 3:
            y_pos = input_blocks[i] + output_blocks[i]/2
            ax.text(i, y_pos, f'{output_blocks[i]:.1f}%',
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)

        # Label for bypassed (if > 3%)
        if bypasses[i] > 3:
            y_pos = input_blocks[i] + output_blocks[i] + bypasses[i]/2
            ax.text(i, y_pos, f'{bypasses[i]:.1f}%',
                    ha='center', va='center', fontweight='bold', color='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'jailbreak_effectiveness.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: jailbreak_effectiveness.png")
    plt.close()


def plot_false_positive_analysis(summaries: Dict[str, Dict], output_dir: Path):
    """Plot false positive rates by configuration and category."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Order: permissive, balanced, strict
    config_order = ['permissive', 'balanced', 'strict']
    configs = [c for c in config_order if c in summaries]

    # Overall false positive rates
    fp_rates = []
    for config in configs:
        if "false_positives" in summaries[config]:
            fp_rates.append(summaries[config]["false_positives"]["false_positive_rate"] * 100)
        else:
            fp_rates.append(0)

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    ax1.bar(range(len(configs)), fp_rates, color=colors[:len(configs)])
    ax1.set_ylabel('False Positive Rate (%)', fontsize=12)
    ax1.set_title('Overall False Positive Rates', fontsize=13, fontweight='bold')
    ax1.set_xticks(range(len(configs)))
    ax1.set_xticklabels([c.replace('_', ' ').title() for c in configs])
    ax1.set_ylim(0, max(fp_rates) * 1.2 if fp_rates else 10)

    # Add value labels
    for i, rate in enumerate(fp_rates):
        ax1.text(i, rate + 0.5, f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Add target line at 5%
    ax1.axhline(y=5, color='red', linestyle='--', alpha=0.7, label='Target (<5%)')
    ax1.legend()

    # By category (use balanced mode as example)
    if configs and "false_positives" in summaries[configs[0]]:
        fp_data = summaries[configs[0]]["false_positives"]
        if "by_category" in fp_data:
            categories = list(fp_data["by_category"].keys())
            fp_by_cat = []

            for cat in categories:
                stats = fp_data["by_category"][cat]
                fp_rate = (stats["false_positives"] / stats["total"] * 100) if stats["total"] > 0 else 0
                fp_by_cat.append(fp_rate)

            # Sort by FP rate
            sorted_indices = np.argsort(fp_by_cat)[::-1]
            categories = [categories[i] for i in sorted_indices]
            fp_by_cat = [fp_by_cat[i] for i in sorted_indices]

            ax2.barh(range(len(categories)), fp_by_cat, color='#3498db')
            ax2.set_xlabel('False Positive Rate (%)', fontsize=12)
            ax2.set_title(f'FP Rates by Category ({configs[0].replace("_", " ").title()})',
                         fontsize=13, fontweight='bold')
            ax2.set_yticks(range(len(categories)))
            ax2.set_yticklabels([c.title() for c in categories])
            ax2.set_xlim(0, max(fp_by_cat) * 1.2 if fp_by_cat else 10)

            # Add value labels
            for i, rate in enumerate(fp_by_cat):
                ax2.text(rate + 0.3, i, f'{rate:.1f}%', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_dir / 'false_positive_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: false_positive_analysis.png")
    plt.close()


def plot_latency_comparison(summaries: Dict[str, Dict], output_dir: Path):
    """Plot latency overhead by configuration."""
    # Order: permissive, balanced, strict
    config_order = ['permissive', 'balanced', 'strict']
    configs = [c for c in config_order if c in summaries]

    # Extract latency metrics
    mean_latencies = []
    p95_latencies = []
    p99_latencies = []

    for config in configs:
        if "latency" in summaries[config]:
            lat = summaries[config]["latency"]
            mean_latencies.append(lat.get("mean_latency_ms", 0))
            p95_latencies.append(lat.get("p95_latency_ms", 0))
            p99_latencies.append(lat.get("p99_latency_ms", 0))
        else:
            mean_latencies.append(0)
            p95_latencies.append(0)
            p99_latencies.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(configs))
    width = 0.25

    ax.bar(x - width, mean_latencies, width, label='Mean', color='#3498db')
    ax.bar(x, p95_latencies, width, label='P95', color='#f39c12')
    ax.bar(x + width, p99_latencies, width, label='P99', color='#e74c3c')

    ax.set_ylabel('Latency (ms)', fontsize=12)
    ax.set_title('Request Latency by Configuration', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c.replace('_', ' ').title() for c in configs])
    ax.legend()

    # Add target line at 200ms
    ax.axhline(y=200, color='red', linestyle='--', alpha=0.7, label='Target (<200ms)')

    plt.tight_layout()
    plt.savefig(output_dir / 'latency_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: latency_comparison.png")
    plt.close()


def plot_security_vs_usability(summaries: Dict[str, Dict], output_dir: Path):
    """Plot security vs usability trade-off curve."""
    # Order: permissive, balanced, strict
    config_order = ['permissive', 'balanced', 'strict']
    configs = [c for c in config_order if c in summaries]

    # Calculate metrics
    security_scores = []  # Total block rate (higher = more secure)
    usability_scores = []  # 100 - FP rate (higher = more usable)

    for config in configs:
        # Security: jailbreak block rate
        if "jailbreak_resistance" in summaries[config]:
            jr = summaries[config]["jailbreak_resistance"]
            security = (jr["block_rate"] + jr["output_catch_rate"]) * 100
        else:
            security = 0

        # Usability: inverse of false positive rate
        if "false_positives" in summaries[config]:
            fp_rate = summaries[config]["false_positives"]["false_positive_rate"] * 100
            usability = 100 - fp_rate
        else:
            usability = 100

        security_scores.append(security)
        usability_scores.append(usability)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    for i, config in enumerate(configs):
        ax.scatter(usability_scores[i], security_scores[i],
                  s=500, alpha=0.6, c=colors[i], label=config.replace('_', ' ').title())
        ax.annotate(config.replace('_', ' ').title(),
                   (usability_scores[i], security_scores[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=11, fontweight='bold')

    ax.set_xlabel('Usability (100 - False Positive Rate %)', fontsize=12)
    ax.set_ylabel('Security (Jailbreak Block Rate %)', fontsize=12)
    ax.set_title('Security vs Usability Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(80, 105)
    ax.set_ylim(0, 105)

    # Add ideal region
    ax.axhspan(75, 100, xmin=0.9, xmax=1.0, alpha=0.1, color='green', label='Ideal Region')
    ax.legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(output_dir / 'security_vs_usability.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: security_vs_usability.png")
    plt.close()


def plot_attack_category_heatmap(summaries: Dict[str, Dict], output_dir: Path):
    """Plot heatmap of block rates by attack category and config."""
    # Order: permissive, balanced, strict
    config_order = ['permissive', 'balanced', 'strict']
    configs = [c for c in config_order if c in summaries]

    # Get all categories
    all_categories = set()
    for config in configs:
        if "jailbreak_resistance" in summaries[config]:
            jr = summaries[config]["jailbreak_resistance"]
            if "by_category" in jr:
                all_categories.update(jr["by_category"].keys())

    categories = sorted(all_categories)

    # Build matrix
    matrix = []
    for category in categories:
        row = []
        for config in configs:
            if "jailbreak_resistance" in summaries[config]:
                jr = summaries[config]["jailbreak_resistance"]
                if "by_category" in jr and category in jr["by_category"]:
                    stats = jr["by_category"][category]
                    block_rate = ((stats["blocked_at_input"] + stats["blocked_at_output"]) /
                                 stats["total"] * 100) if stats["total"] > 0 else 0
                    row.append(block_rate)
                else:
                    row.append(0)
            else:
                row.append(0)
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    ax.set_xticks(np.arange(len(configs)))
    ax.set_yticks(np.arange(len(categories)))
    ax.set_xticklabels([c.replace('_', ' ').title() for c in configs])
    ax.set_yticklabels([c.replace('_', ' ').title() for c in categories])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(len(categories)):
        for j in range(len(configs)):
            text = ax.text(j, i, f'{matrix[i][j]:.0f}%',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')

    ax.set_title('Attack Block Rate by Category and Configuration', fontsize=14, fontweight='bold', pad=20)
    fig.colorbar(im, ax=ax, label='Block Rate (%)')

    plt.tight_layout()
    plt.savefig(output_dir / 'attack_category_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: attack_category_heatmap.png")
    plt.close()


def plot_radar_charts(summaries: Dict[str, Dict], output_dir: Path):
    """Plot radar charts showing block rates by category for each configuration."""
    # Order: permissive, balanced, strict
    config_order = ['permissive', 'balanced', 'strict']
    configs = [c for c in config_order if c in summaries]

    # Get all categories
    all_categories = set()
    for config in configs:
        if "jailbreak_resistance" in summaries[config]:
            jr = summaries[config]["jailbreak_resistance"]
            if "by_category" in jr:
                all_categories.update(jr["by_category"].keys())

    categories = sorted(all_categories)
    num_categories = len(categories)
    num_configs = len(configs)

    # Calculate angles for each category
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle

    # Create subplots in a single row
    fig, axes = plt.subplots(1, num_configs, figsize=(6 * num_configs, 6),
                             subplot_kw=dict(projection='polar'))
    if num_configs == 1:
        axes = [axes]  # Make it iterable

    # Colors for each defense layer
    colors = {
        "input": "#2ecc71",      # Green
        "output": "#f39c12",     # Orange
        "bypassed": "#e74c3c"    # Red
    }

    for idx, config in enumerate(configs):
        ax = axes[idx]

        # Calculate block rates for each category
        input_rates = []
        output_rates = []
        bypass_rates = []

        for category in categories:
            if "jailbreak_resistance" in summaries[config]:
                jr = summaries[config]["jailbreak_resistance"]
                if "by_category" in jr and category in jr["by_category"]:
                    stats = jr["by_category"][category]
                    total = stats["total"]

                    if total > 0:
                        input_rate = (stats["blocked_at_input"] / total) * 100
                        output_rate = (stats["blocked_at_output"] / total) * 100
                        bypass_rate = (stats["allowed"] / total) * 100

                        input_rates.append(input_rate)
                        output_rates.append(output_rate)
                        bypass_rates.append(bypass_rate)
                    else:
                        input_rates.append(0)
                        output_rates.append(0)
                        bypass_rates.append(0)
                else:
                    input_rates.append(0)
                    output_rates.append(0)
                    bypass_rates.append(0)
            else:
                input_rates.append(0)
                output_rates.append(0)
                bypass_rates.append(0)

        # Close the polygons
        input_rates += input_rates[:1]
        output_rates += output_rates[:1]
        bypass_rates += bypass_rates[:1]

        # Create stacked rings from center outward (like Exp 04)
        center = [0] * len(angles)
        input_outer = input_rates[:]
        output_outer = [(i + o) for i, o in zip(input_rates, output_rates)]
        bypass_outer = [100] * len(angles)

        # Plot stacked areas from inside to outside
        ax.fill_between(angles, center, input_outer,
                        alpha=0.7, color=colors["input"], label='Input Blocked')
        ax.fill_between(angles, input_outer, output_outer,
                        alpha=0.7, color=colors["output"], label='Output Blocked')
        ax.fill_between(angles, output_outer, bypass_outer,
                        alpha=0.7, color=colors["bypassed"], label='Bypassed')

        # Add edge lines for clarity
        ax.plot(angles, input_outer, 'o-', linewidth=1.5, color='#27ae60', alpha=0.8, markersize=4)
        ax.plot(angles, output_outer, 'o-', linewidth=1.5, color='#e67e22', alpha=0.8, markersize=4)

        # Customize
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([c.replace('_', ' ').title() for c in categories], size=10)
        ax.set_ylim(0, 100)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=9.5)
        ax.grid(True, linestyle='-', alpha=0.5, linewidth=0.7, color='gray')

        # Title with defense summary
        if "jailbreak_resistance" in summaries[config]:
            jr = summaries[config]["jailbreak_resistance"]
            total_tests = jr["total"]
            input_blocks = jr["blocked_at_input"]
            output_blocks = jr["blocked_at_output"]
            allowed = jr["allowed"]

            # Set config name as bold text at top
            config_name = config.replace('_', ' ').title()
            ax.text(0.5, 1.17, config_name, transform=ax.transAxes,
                    ha='center', va='top', fontsize=12, fontweight='bold')

            # Add stats below the config name
            stats_text = f"In:{input_blocks}/{total_tests} ({input_blocks/total_tests*100:.1f}%) "
            stats_text += f"Out:{output_blocks} ({output_blocks/total_tests*100:.1f}%) "
            stats_text += f"Bypass:{allowed} ({allowed/total_tests*100:.1f}%)"
            ax.text(0.5, 1.12, stats_text, transform=ax.transAxes,
                    ha='center', va='top', fontsize=9)

    # Add legend to the figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02),
               ncol=3, frameon=True, fontsize=11)

    plt.tight_layout()
    plt.savefig(output_dir / 'radar_charts.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: radar_charts.png")
    plt.close()


def generate_markdown_report(summaries: Dict[str, Dict], output_dir: Path):
    """Generate markdown analysis report."""
    report = ["# Guardrail Testing Analysis Report\n"]
    report.append(f"Generated: {Path(output_dir).absolute()}\n")
    report.append("---\n\n")

    for config in summaries.keys():
        report.append(f"## Configuration: {config.replace('_', ' ').title()}\n\n")

        summary = summaries[config]

        # Jailbreak Resistance
        if "jailbreak_resistance" in summary:
            jr = summary["jailbreak_resistance"]
            report.append("### Jailbreak Resistance\n\n")
            report.append(f"- **Total Tests**: {jr['total']}\n")
            report.append(f"- **Blocked at Input**: {jr['blocked_at_input']} ({jr['block_rate']*100:.1f}%)\n")
            report.append(f"- **Blocked at Output**: {jr['blocked_at_output']} ({jr['output_catch_rate']*100:.1f}%)\n")
            report.append(f"- **Bypassed**: {jr['allowed']} ({jr['bypass_rate']*100:.1f}%)\n\n")

        # False Positives
        if "false_positives" in summary:
            fp = summary["false_positives"]
            report.append("### False Positives\n\n")
            report.append(f"- **Total Tests**: {fp['total']}\n")
            report.append(f"- **False Positives**: {fp['false_positives']} ({fp['false_positive_rate']*100:.1f}%)\n")
            report.append(f"- **Correctly Allowed**: {fp['allowed']}\n\n")

        # Latency
        if "latency" in summary:
            lat = summary["latency"]
            report.append("### Latency Performance\n\n")
            report.append(f"- **Mean**: {lat['mean_latency_ms']:.1f} ms\n")
            report.append(f"- **Median**: {lat['median_latency_ms']:.1f} ms\n")
            report.append(f"- **P95**: {lat['p95_latency_ms']:.1f} ms\n")
            report.append(f"- **P99**: {lat['p99_latency_ms']:.1f} ms\n\n")

        report.append("---\n\n")

    # Save report
    report_file = output_dir / 'analysis_report.md'
    with open(report_file, 'w') as f:
        f.writelines(report)

    print(f"✓ Saved: analysis_report.md")


def main():
    parser = argparse.ArgumentParser(description="Analyse guardrail test results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory containing test results"
    )

    args = parser.parse_args()

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    results_dir = script_dir / args.results_dir

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load all summary files
    summary_files = list(results_dir.glob("summary_*.json"))

    if not summary_files:
        print(f"Error: No summary files found in {results_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Analysing Guardrail Test Results")
    print(f"Found {len(summary_files)} configuration(s)")
    print(f"{'='*60}\n")

    # Load summaries
    summaries = {}
    for summary_file in summary_files:
        config_name = summary_file.stem.replace('summary_', '')
        summaries[config_name] = load_summary(summary_file)
        print(f"✓ Loaded: {config_name}")

    # Create visualizations
    print(f"\nGenerating visualizations...")

    plot_jailbreak_effectiveness(summaries, results_dir)
    plot_false_positive_analysis(summaries, results_dir)
    plot_latency_comparison(summaries, results_dir)
    plot_security_vs_usability(summaries, results_dir)
    plot_attack_category_heatmap(summaries, results_dir)
    plot_radar_charts(summaries, results_dir)

    # Generate report
    generate_markdown_report(summaries, results_dir)

    print(f"\n{'='*60}")
    print(f"Analysis Complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
