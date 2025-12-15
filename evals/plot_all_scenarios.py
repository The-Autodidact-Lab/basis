#!/usr/bin/env python3
"""
Generate comprehensive plots for all scenarios with pass^k, pass@k, and success rate.

Creates a 3x3 grid showing:
- Row 1: premium_bias
- Row 2: cab_quote_only_vs_book  
- Row 3: cab_stale_locations
- Column 1: Pass^k
- Column 2: Pass@k
- Column 3: Raw Success Rate
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available - plots will be skipped")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
SCENARIOS = [
    "premium_bias",
    "cab_quote_only_vs_book",
    "cab_stale_locations",
]

AGENTS = {
    "default_multi_agent": "without Cortex",
    "multi_agent": "with Cortex",
}

PASS_AT_K_VALUES = [1, 2, 5, 10]


def extract_results_from_directory(pass_k_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract all results from a pass_at_k directory.
    Handles set-based structure: set_1/run_1/, set_1/run_2/, etc.
    """
    results = []
    
    # Look for set directories
    for set_dir in sorted(pass_k_dir.iterdir()):
        if not set_dir.is_dir() or not set_dir.name.startswith("set_"):
            continue
        
        # Extract set number
        try:
            set_number = int(set_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        
        # Look for run directories within this set
        for run_dir in sorted(set_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            
            # Extract run number
            try:
                run_number = int(run_dir.name.split("_")[1])
            except (ValueError, IndexError):
                continue
            
            output_jsonl = run_dir / "output.jsonl"
            if not output_jsonl.exists():
                continue
            
            try:
                with open(output_jsonl, "r") as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            task_id = data.get("task_id")
                            score = data.get("score", 0.0)
                            metadata = data.get("metadata", {})
                            scenario_id = metadata.get("scenario_id", task_id)
                            
                            results.append({
                                "task_id": task_id,
                                "scenario_id": scenario_id,
                                "set_number": set_number,
                                "run_number": run_number,
                                "score": score,
                                "success": score > 0.0,
                                "status": metadata.get("status", "unknown"),
                                "metadata": metadata,
                            })
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Failed to read {output_jsonl}: {e}")
                continue
    
    return results


def calculate_pass_hat_k(results: List[Dict[str, Any]], k: int) -> float:
    """
    Calculate pass^k: probability that ALL k runs succeed.
    """
    if not results:
        return 0.0
    
    # Group results by set_number
    set_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        set_number = result.get("set_number")
        if set_number is not None:
            set_results[set_number].append(result)
    
    if not set_results:
        return 0.0
    
    # Sort runs by run_number within each set
    for set_number in set_results:
        runs = set_results[set_number]
        runs.sort(key=lambda x: x.get("run_number", 999))
    
    # Calculate pass^k for each set
    sets_with_all_success = 0
    total_sets = len(set_results)
    
    for set_number, runs in set_results.items():
        set_runs = runs[:k]
        if len(set_runs) < k:
            continue
        if all(run.get("success", False) for run in set_runs):
            sets_with_all_success += 1
    
    return sets_with_all_success / total_sets if total_sets > 0 else 0.0


def calculate_pass_at_k(results: List[Dict[str, Any]], k: int) -> float:
    """
    Calculate pass@k: probability that AT LEAST ONE of k runs succeeds.
    """
    if not results:
        return 0.0
    
    # Group results by set_number
    set_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        set_number = result.get("set_number")
        if set_number is not None:
            set_results[set_number].append(result)
    
    if not set_results:
        return 0.0
    
    # Sort runs by run_number within each set
    for set_number in set_results:
        runs = set_results[set_number]
        runs.sort(key=lambda x: x.get("run_number", 999))
    
    # Calculate pass@k for each set
    sets_with_at_least_one_success = 0
    total_sets = len(set_results)
    
    for set_number, runs in set_results.items():
        set_runs = runs[:k]
        if len(set_runs) < k:
            continue
        if any(run.get("success", False) for run in set_runs):
            sets_with_at_least_one_success += 1
    
    return sets_with_at_least_one_success / total_sets if total_sets > 0 else 0.0


def calculate_success_rate(results: List[Dict[str, Any]]) -> float:
    """
    Calculate raw success rate across all runs.
    """
    if not results:
        return 0.0
    
    total_runs = len(results)
    successful_runs = sum(1 for r in results if r.get("success", False))
    
    return successful_runs / total_runs if total_runs > 0 else 0.0


def analyze_all_scenarios(base_output_dir: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Analyze all scenarios and calculate pass^k, pass@k, and success rate.
    
    Returns: {scenario_id: {agent: {metric: value}}}
    """
    logger.info("=" * 80)
    logger.info("Analyzing All Scenarios")
    logger.info("=" * 80)
    
    analysis = {}
    
    for scenario_id in SCENARIOS:
        logger.info(f"\n  Scenario: {scenario_id}")
        analysis[scenario_id] = {}
        
        for agent, agent_label in AGENTS.items():
            logger.info(f"    Agent: {agent_label}")
            analysis[scenario_id][agent] = {
                "pass_hat_k": {},
                "pass_at_k": {},
                "success_rate": {},
            }
            
            agent_dir = base_output_dir / scenario_id / agent
            
            if not agent_dir.exists():
                logger.warning(f"      Directory not found: {agent_dir}")
                continue
            
            # Collect all results across all k values for success rate calculation
            all_results = []
            
            for k in PASS_AT_K_VALUES:
                pass_k_dir = agent_dir / f"pass_at_{k}"
                
                if not pass_k_dir.exists():
                    logger.warning(f"      Directory not found: {pass_k_dir}")
                    analysis[scenario_id][agent]["pass_hat_k"][k] = 0.0
                    analysis[scenario_id][agent]["pass_at_k"][k] = 0.0
                    continue
                
                # Extract results
                results = extract_results_from_directory(pass_k_dir)
                all_results.extend(results)
                
                # Calculate metrics
                pass_hat_k_score = calculate_pass_hat_k(results, k)
                pass_at_k_score = calculate_pass_at_k(results, k)
                
                analysis[scenario_id][agent]["pass_hat_k"][k] = pass_hat_k_score
                analysis[scenario_id][agent]["pass_at_k"][k] = pass_at_k_score
            
            # Calculate overall success rate
            success_rate = calculate_success_rate(all_results)
            analysis[scenario_id][agent]["success_rate"]["overall"] = success_rate
            
            logger.info(f"      Overall success rate: {success_rate*100:.1f}% ({sum(1 for r in all_results if r.get('success', False))}/{len(all_results)} runs)")
    
    return analysis


def create_combined_plot(analysis: Dict[str, Dict[str, Dict[str, float]]], output_dir: Path):
    """
    Create a combined figure:
    - Top row: Pass^k for each of 3 scenarios (horizontal sequence)
    - Bottom row: Pass@k for each of 3 scenarios (horizontal sequence)
    - Right side: Success rate bar chart (spans both rows)
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available - skipping plot generation")
        return
    
    logger.info("\nCreating combined plot...")
    
    # Set clean sans-serif font with bold defaults
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'Verdana']
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    
    # Create figure with custom layout: 2 rows (pass^k, pass@k) x 4 columns (3 scenarios + 1 bar chart)
    fig = plt.figure(figsize=(22, 10))
    gs = fig.add_gridspec(2, 4, hspace=0.4, wspace=0.4, 
                          left=0.06, right=0.98, top=0.92, bottom=0.12)
    
    # Blue gradient color scheme
    colors = {
        "without Cortex": "#5B9BD5",      # Medium blue
        "with Cortex": "#2E75B6",         # Darker blue
    }
    
    # Gradient fills for bars
    bar_colors = {
        "without Cortex": "#7DB4D9",      # Lighter blue
        "with Cortex": "#4472C4",         # Medium-dark blue
    }
    
    line_styles = {
        "without Cortex": "--",
        "with Cortex": "-",
    }
    
    markers = {
        "without Cortex": "o",
        "with Cortex": "s",
    }
    
    # Main title (lighter weight)
    # fig.suptitle('Ablation Study Results: Pass^k, Pass@k, and Success Rate', 
    #              fontsize=16, fontweight='normal', y=0.96)
    
    # Top row: Pass^k for each scenario
    for col_idx, scenario_id in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[0, col_idx])
        
        for agent, agent_label in AGENTS.items():
            if scenario_id not in analysis or agent not in analysis[scenario_id]:
                continue
            
            k_values = []
            pass_hat_k_scores = []
            
            for k in PASS_AT_K_VALUES:
                if k in analysis[scenario_id][agent]["pass_hat_k"]:
                    k_values.append(k)
                    pass_hat_k_scores.append(analysis[scenario_id][agent]["pass_hat_k"][k] * 100)
            
            if k_values:
                ax.plot(k_values, pass_hat_k_scores,
                       marker=markers[agent_label],
                       label=agent_label,
                       linewidth=2.2,
                       markersize=6.5,
                       linestyle=line_styles[agent_label],
                       color=colors[agent_label],
                       alpha=0.85)
        
        scenario_title = scenario_id.replace("_", " ").title()
        ax.set_xlabel("k", fontsize=22, fontweight='bold', color='#000000')
        ax.set_ylabel("Pass^k (%)", fontsize=22, fontweight='bold', color='#000000')
        ax.set_title(scenario_title, fontsize=24, fontweight='bold', pad=10, color='#000000')
        ax.set_xticks(PASS_AT_K_VALUES)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.6, color='#D0D0D0')
        ax.legend(fontsize=24, loc='best', framealpha=0.9, frameon=True, prop={'weight': 'bold'})
        ax.set_facecolor('#FAFAFA')
        ax.tick_params(axis='both', labelsize=18, colors='#000000')
        # Rotate x-axis labels if needed for better spacing
        ax.tick_params(axis='x', rotation=0)
    
    # Bottom row: Pass@k for each scenario
    for col_idx, scenario_id in enumerate(SCENARIOS):
        ax = fig.add_subplot(gs[1, col_idx])
        
        for agent, agent_label in AGENTS.items():
            if scenario_id not in analysis or agent not in analysis[scenario_id]:
                continue
            
            k_values = []
            pass_at_k_scores = []
            
            for k in PASS_AT_K_VALUES:
                if k in analysis[scenario_id][agent]["pass_at_k"]:
                    k_values.append(k)
                    pass_at_k_scores.append(analysis[scenario_id][agent]["pass_at_k"][k] * 100)
            
            if k_values:
                ax.plot(k_values, pass_at_k_scores,
                       marker=markers[agent_label],
                       label=agent_label,
                       linewidth=2.2,
                       markersize=6.5,
                       linestyle=line_styles[agent_label],
                       color=colors[agent_label],
                       alpha=0.85)
        
        scenario_title = scenario_id.replace("_", " ").title()
        ax.set_xlabel("k", fontsize=22, fontweight='bold', color='#000000')
        ax.set_ylabel("Pass@k (%)", fontsize=22, fontweight='bold', color='#000000')
        ax.set_title(scenario_title, fontsize=24, fontweight='bold', pad=10, color='#000000')
        ax.set_xticks(PASS_AT_K_VALUES)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.6, color='#D0D0D0')
        ax.legend(fontsize=24, loc='best', framealpha=0.9, frameon=True, prop={'weight': 'bold'})
        ax.set_facecolor('#FAFAFA')
        ax.tick_params(axis='both', labelsize=18, colors='#000000')
        ax.tick_params(axis='x', rotation=0)
    
    # Right side: Success Rate bar chart (spans both rows)
    ax_success = fig.add_subplot(gs[:, 3])
    
    scenario_labels = [s.replace("_", " ").title() for s in SCENARIOS]
    x = np.arange(len(scenario_labels))
    width = 0.4
    
    without_cortex_rates = []
    with_cortex_rates = []
    
    for scenario_id in SCENARIOS:
        without_rate = analysis[scenario_id].get("default_multi_agent", {}).get("success_rate", {}).get("overall", 0.0) * 100
        with_rate = analysis[scenario_id].get("multi_agent", {}).get("success_rate", {}).get("overall", 0.0) * 100
        without_cortex_rates.append(without_rate)
        with_cortex_rates.append(with_rate)
    
    bars1 = ax_success.bar(x - width/2, without_cortex_rates, width,
                          label='without Cortex',
                          color=bar_colors["without Cortex"],
                          alpha=0.8,
                          edgecolor='#1A4A7A',
                          linewidth=1.5)
    bars2 = ax_success.bar(x + width/2, with_cortex_rates, width,
                          label='with Cortex',
                          color=bar_colors["with Cortex"],
                          alpha=0.8,
                          edgecolor='#0F2D4F',
                          linewidth=1.5)
    
    ax_success.set_ylabel('Success Rate (%)', fontsize=28, fontweight='bold', color='#000000')
    ax_success.set_title('Overall Success Rate\nAcross All Runs', fontsize=24, fontweight='bold', pad=15, color='#000000')
    ax_success.set_xticks(x)
    ax_success.set_xticklabels(scenario_labels, fontsize=22, fontweight='bold', rotation=60, ha='right', va='top', color='#000000')
    ax_success.set_ylim([0, 105])
    ax_success.legend(fontsize=24, loc='upper left', framealpha=0.9, frameon=True, prop={'weight': 'bold'})
    ax_success.grid(True, alpha=0.2, linestyle='-', linewidth=0.6, color='#D0D0D0', axis='y')
    ax_success.set_facecolor('#FAFAFA')
    ax_success.tick_params(axis='y', labelsize=24, colors='#000000')
    ax_success.tick_params(axis='x', labelsize=22, colors='#000000')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax_success.text(bar.get_x() + bar.get_width()/2., height + 2,
                           f'{height:.1f}%',
                           ha='center', va='bottom', fontsize=18, fontweight='bold',
                           color='#000000')
    
    # Save the combined plot
    plot_path = output_dir / "all_scenarios_combined.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    logger.info(f"✓ Saved combined plot to {plot_path}")
    plt.close()
    
    # Also create individual metric plots (3 separate plots, one for each metric)
    for metric_idx, (metric_name, metric_key, ylabel) in enumerate([
        ("pass_hat_k", "pass_hat_k", "Pass^k (%)"),
        ("pass_at_k", "pass_at_k", "Pass@k (%)"),
    ]):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f'{ylabel.replace(" (%)", "")} Across All Scenarios', 
                     fontsize=24, fontweight='bold', y=1.02, color='#000000')
        
        for col_idx, scenario_id in enumerate(SCENARIOS):
            ax = axes[col_idx]
            
            for agent, agent_label in AGENTS.items():
                if scenario_id not in analysis or agent not in analysis[scenario_id]:
                    continue
                
                k_values = []
                scores = []
                
                for k in PASS_AT_K_VALUES:
                    if k in analysis[scenario_id][agent][metric_key]:
                        k_values.append(k)
                        scores.append(analysis[scenario_id][agent][metric_key][k] * 100)
                
                if k_values:
                    ax.plot(k_values, scores,
                           marker=markers[agent_label],
                           label=agent_label,
                           linewidth=2.5,
                           markersize=8,
                           linestyle=line_styles[agent_label],
                           color=colors[agent_label],
                           alpha=0.9)
            
            ax.set_xlabel("k", fontsize=22, fontweight='bold', color='#000000')
            ax.set_ylabel(ylabel, fontsize=22, fontweight='bold', color='#000000')
            ax.set_title(scenario_id.replace("_", " ").title(), fontsize=24, fontweight='bold', color='#000000')
            ax.set_xticks(PASS_AT_K_VALUES)
            ax.set_ylim([0, 105])
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
            ax.legend(fontsize=24, loc='best', framealpha=0.9, prop={'weight': 'bold'})
            ax.tick_params(axis='both', labelsize=18, colors='#000000')
        
        plt.tight_layout()
        metric_plot_path = output_dir / f"all_scenarios_{metric_name}.png"
        plt.savefig(metric_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"✓ Saved {metric_name} plot to {metric_plot_path}")
    
    # Success rate comparison plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scenario_labels = [s.replace("_", " ").title() for s in SCENARIOS]
    x = np.arange(len(scenario_labels))
    width = 0.35
    
    without_cortex_rates = []
    with_cortex_rates = []
    
    for scenario_id in SCENARIOS:
        without_rate = analysis[scenario_id].get("default_multi_agent", {}).get("success_rate", {}).get("overall", 0.0) * 100
        with_rate = analysis[scenario_id].get("multi_agent", {}).get("success_rate", {}).get("overall", 0.0) * 100
        without_cortex_rates.append(without_rate)
        with_cortex_rates.append(with_rate)
    
    bars1 = ax.bar(x - width/2, without_cortex_rates, width, label='without Cortex', 
                   color=colors["without Cortex"], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, with_cortex_rates, width, label='with Cortex',
                   color=colors["with Cortex"], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Success Rate (%)', fontsize=28, fontweight='bold', color='#000000')
    ax.set_title('Overall Success Rate Comparison Across Scenarios', fontsize=24, fontweight='bold', pad=15, color='#000000')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=22, fontweight='bold', rotation=60, ha='right', va='top', color='#000000')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=24, loc='best', framealpha=0.9, prop={'weight': 'bold'})
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.8, axis='y')
    ax.tick_params(axis='y', labelsize=24, colors='#000000')
    ax.tick_params(axis='x', labelsize=22, colors='#000000')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=18, fontweight='bold', color='#000000')
    
    plt.tight_layout()
    success_rate_plot_path = output_dir / "all_scenarios_success_rate.png"
    plt.savefig(success_rate_plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    logger.info(f"✓ Saved success rate plot to {success_rate_plot_path}")


def print_summary(analysis: Dict[str, Dict[str, Dict[str, float]]]):
    """Print a comprehensive summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 80)
    
    for scenario_id in SCENARIOS:
        logger.info(f"\n{scenario_id.replace('_', ' ').title()}:")
        for agent, agent_label in AGENTS.items():
            if scenario_id not in analysis or agent not in analysis[scenario_id]:
                continue
            
            logger.info(f"  {agent_label}:")
            
            # Pass^k
            logger.info(f"    Pass^k:")
            for k in PASS_AT_K_VALUES:
                score = analysis[scenario_id][agent]["pass_hat_k"].get(k, 0.0) * 100
                logger.info(f"      k={k}: {score:.1f}%")
            
            # Pass@k
            logger.info(f"    Pass@k:")
            for k in PASS_AT_K_VALUES:
                score = analysis[scenario_id][agent]["pass_at_k"].get(k, 0.0) * 100
                logger.info(f"      k={k}: {score:.1f}%")
            
            # Success rate
            success_rate = analysis[scenario_id][agent]["success_rate"].get("overall", 0.0) * 100
            logger.info(f"    Overall Success Rate: {success_rate:.1f}%")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate comprehensive plots for all scenarios"
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./evals/output_07dec_pass^k_small",
        help="Base directory containing the results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: same as input_dir)",
    )
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1
    
    # Analyze all scenarios
    analysis = analyze_all_scenarios(input_dir)
    
    # Print summary
    print_summary(analysis)
    
    # Save analysis
    analysis_file = output_dir / "all_scenarios_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"\n✓ Analysis saved to: {analysis_file}")
    
    # Create plots
    try:
        create_combined_plot(analysis, output_dir)
        logger.info("\n✓ All plots generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate plots: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1
    
    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

