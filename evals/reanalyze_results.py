#!/usr/bin/env python3
"""
Re-analyze ablation study results with detailed statistics and correct pass^k calculation.

This script provides comprehensive analysis including:
- Number of trials per scenario/agent/k combination
- Number of successes
- Pass^k scores
- Detailed breakdowns
"""

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Scenarios and agents
SCENARIOS = [
    "case_1_premium_bias",
    "cab_quote_only_vs_book",
    "cab_stale_locations",
]

AGENTS = {
    "default_multi_agent": "Control (default_multi_agent)",
    "multi_agent": "Experimental (multi_agent)",
}

PASS_AT_K_VALUES = [1, 2, 5, 10, 20]


def extract_run_number_from_path(file_path: Path) -> int | None:
    """Extract run number from a file path like .../run_3/..."""
    parts = file_path.parts
    for part in parts:
        if part.startswith("run_") and part[4:].isdigit():
            return int(part[4:])
    return None


def extract_results_from_directory(pass_k_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract all results from a pass_at_k directory.
    Handles both new set-based structure (set_1/run_1/) and old structure (run_1/).
    Returns results with set_number and run_number extracted from directory structure.
    """
    results = []
    
    # Look for set directories (new structure)
    for set_dir in sorted(pass_k_dir.iterdir()):
        if not set_dir.is_dir():
            continue
        
        # Extract set number if it's a set directory
        set_number = None
        if set_dir.name.startswith("set_"):
            try:
                set_number = int(set_dir.name.split("_")[1])
            except (ValueError, IndexError):
                pass
        
        # Look for run directories
        for run_dir in sorted(set_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            
            # Extract run number from directory name
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
                logger.warning(f"Failed to read {output_jsonl}: {e}")
    
    # Fallback: look for run_* directories directly (old structure, backward compatibility)
    if not results:
        for run_dir in sorted(pass_k_dir.iterdir()):
            if not run_dir.is_dir() or not run_dir.name.startswith("run_"):
                continue
            
            run_number = extract_run_number_from_path(run_dir)
            if run_number is None:
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
                                "set_number": None,  # No set in old structure
                                "run_number": run_number,
                                "score": score,
                                "success": score > 0.0,
                                "status": metadata.get("status", "unknown"),
                                "metadata": metadata,
                            })
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to read {output_jsonl}: {e}")
    
    return results


def calculate_pass_hat_k_detailed(results: List[Dict[str, Any]], k: int) -> Dict[str, Any]:
    """
    Calculate pass^k with detailed statistics across sets.
    
    Returns a dictionary with:
    - pass_hat_k: The pass^k score (average across sets)
    - total_sets: Number of unique sets
    - sets_with_all_success: Number of sets where all k runs succeeded
    - total_runs: Total number of runs
    - successful_runs: Number of successful runs
    - run_details: List of set details for debugging
    """
    if not results:
        return {
            "pass_hat_k": 0.0,
            "total_sets": 0,
            "sets_with_all_success": 0,
            "total_runs": 0,
            "successful_runs": 0,
            "success_rate": 0.0,
            "run_details": [],
        }
    
    # Group results by set_number (new structure) or scenario_id (old structure)
    set_results: Dict[int | str, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        set_number = result.get("set_number")
        if set_number is not None:
            set_results[set_number].append(result)
        else:
            # Fallback to scenario_id for backward compatibility
            scenario_id = result.get("scenario_id") or result.get("task_id", "unknown")
            set_results[scenario_id].append(result)
    
    # Sort runs by run_number within each set
    for set_key in set_results:
        runs = set_results[set_key]
        runs.sort(key=lambda x: x.get("run_number", 999))
    
    # Calculate statistics
    total_sets = len(set_results)
    sets_with_all_success = 0
    total_runs = 0
    successful_runs = 0
    run_details = []
    
    for set_key, runs in set_results.items():
        # Take only the first k runs for this set
        set_runs = runs[:k]
        total_runs += len(set_runs)
        
        # Count successes
        for run in set_runs:
            if run.get("success", False):
                successful_runs += 1
        
        # Check if ALL k runs succeeded
        if len(set_runs) >= k:
            all_succeeded = all(run.get("success", False) for run in set_runs)
            if all_succeeded:
                sets_with_all_success += 1
            
            # Store details for this set
            run_details.append({
                "set_number": set_key if isinstance(set_key, int) else None,
                "scenario_id": set_key if isinstance(set_key, str) else runs[0].get("scenario_id") if runs else None,
                "num_runs": len(set_runs),
                "num_successes": sum(1 for r in set_runs if r.get("success", False)),
                "all_succeeded": all_succeeded,
                "runs": [
                    {
                        "run_number": r.get("run_number"),
                        "success": r.get("success", False),
                        "score": r.get("score", 0.0),
                        "status": r.get("status", "unknown"),
                    }
                    for r in set_runs
                ],
            })
    
    pass_hat_k = sets_with_all_success / total_sets if total_sets > 0 else 0.0
    
    return {
        "pass_hat_k": pass_hat_k,
        "total_sets": total_sets,
        "sets_with_all_success": sets_with_all_success,
        "total_scenarios": total_sets,  # For backward compatibility
        "scenarios_with_all_success": sets_with_all_success,  # For backward compatibility
        "total_runs": total_runs,
        "successful_runs": successful_runs,
        "success_rate": successful_runs / total_runs if total_runs > 0 else 0.0,
        "run_details": run_details,
    }


def analyze_all_results(base_output_dir: Path) -> Dict[str, Any]:
    """
    Analyze all results with detailed statistics.
    """
    logger.info("=" * 80)
    logger.info("Re-analyzing Results with Detailed Statistics")
    logger.info("=" * 80)
    
    analysis = {}
    detailed_stats = {}
    
    for scenario_id in SCENARIOS:
        logger.info(f"\n{'='*80}")
        logger.info(f"Scenario: {scenario_id}")
        logger.info(f"{'='*80}")
        
        analysis[scenario_id] = {}
        detailed_stats[scenario_id] = {}
        
        for agent, agent_label in AGENTS.items():
            logger.info(f"\n  Agent: {agent_label}")
            logger.info(f"  {'-'*76}")
            
            analysis[scenario_id][agent] = {}
            detailed_stats[scenario_id][agent] = {}
            
            agent_dir = base_output_dir / scenario_id / agent
            
            if not agent_dir.exists():
                logger.warning(f"    Directory not found: {agent_dir}")
                continue
            
            for k in PASS_AT_K_VALUES:
                pass_k_dir = agent_dir / f"pass_at_{k}"
                
                if not pass_k_dir.exists():
                    logger.warning(f"    Directory not found: {pass_k_dir}")
                    analysis[scenario_id][agent][f"pass_hat_{k}"] = 0.0
                    detailed_stats[scenario_id][agent][f"pass_hat_{k}"] = {
                        "pass_hat_k": 0.0,
                        "total_scenarios": 0,
                        "scenarios_with_all_success": 0,
                        "total_runs": 0,
                        "successful_runs": 0,
                        "success_rate": 0.0,
                    }
                    continue
                
                # Extract results
                all_results = extract_results_from_directory(pass_k_dir)
                
                # Calculate detailed statistics
                stats = calculate_pass_hat_k_detailed(all_results, k)
                
                analysis[scenario_id][agent][f"pass_hat_{k}"] = stats["pass_hat_k"]
                detailed_stats[scenario_id][agent][f"pass_hat_{k}"] = stats
                
                # Print detailed information
                logger.info(f"\n    Pass^{k}:")
                logger.info(f"      Score: {stats['pass_hat_k']:.3f} ({stats['pass_hat_k']*100:.1f}%)")
                total_sets = stats.get('total_sets', stats.get('total_scenarios', 0))
                sets_with_success = stats.get('sets_with_all_success', stats.get('scenarios_with_all_success', 0))
                logger.info(f"      Sets: {total_sets} total, {sets_with_success} with all {k} runs successful")
                logger.info(f"      Runs: {stats['total_runs']} total, {stats['successful_runs']} successful ({stats['success_rate']*100:.1f}% success rate)")
                
                # Show run details for first set if available
                if stats['run_details']:
                    detail = stats['run_details'][0]
                    set_info = f"set {detail['set_number']}" if detail.get('set_number') is not None else f"scenario '{detail.get('scenario_id', 'unknown')}'"
                    logger.info(f"      Example {set_info}:")
                    logger.info(f"        Runs analyzed: {detail['num_runs']}/{k} (need {k} for pass^{k})")
                    logger.info(f"        Successful runs: {detail['num_successes']}/{detail['num_runs']}")
                    logger.info(f"        All {k} succeeded: {detail['all_succeeded']}")
                    if detail['runs']:
                        logger.info(f"        Run breakdown:")
                        for run_info in detail['runs']:
                            status_icon = "✓" if run_info['success'] else "✗"
                            logger.info(f"          {status_icon} Run {run_info['run_number']}: score={run_info['score']:.1f}, status={run_info['status']}")
    
    return analysis, detailed_stats


def print_summary_table(analysis: Dict[str, Any], detailed_stats: Dict[str, Any]):
    """Print a comprehensive summary table."""
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY TABLE")
    logger.info("=" * 80)
    
    # Header
    header_parts = [f"{'Scenario':<25}", f"{'Agent':<25}"]
    for k in PASS_AT_K_VALUES:
        header_parts.append(f"Pass^{k:>2}")
    header = " ".join(header_parts)
    logger.info(header)
    logger.info("-" * len(header))
    
    for scenario_id in SCENARIOS:
        for agent, agent_label in AGENTS.items():
            row = f"{scenario_id:<25} {agent_label:<25} "
            for k in PASS_AT_K_VALUES:
                key = f"pass_hat_{k}"
                if scenario_id in analysis and agent in analysis[scenario_id]:
                    score = analysis[scenario_id][agent].get(key, 0.0) * 100
                    row += f"{score:>6.1f}%  "
                else:
                    row += f"{'N/A':>8}  "
            logger.info(row)
    
    # Detailed statistics summary
    logger.info("\n" + "=" * 80)
    logger.info("DETAILED STATISTICS")
    logger.info("=" * 80)
    
    for scenario_id in SCENARIOS:
        logger.info(f"\n{scenario_id}:")
        for agent, agent_label in AGENTS.items():
            logger.info(f"  {agent_label}:")
            for k in PASS_AT_K_VALUES:
                key = f"pass_hat_{k}"
            if scenario_id in detailed_stats and agent in detailed_stats[scenario_id]:
                stats = detailed_stats[scenario_id][agent].get(key, {})
                total_sets = stats.get('total_sets', stats.get('total_scenarios', 0))
                sets_with_success = stats.get('sets_with_all_success', stats.get('scenarios_with_all_success', 0))
                logger.info(f"    Pass^{k}: {stats.get('pass_hat_k', 0.0)*100:.1f}% "
                          f"({sets_with_success}/{total_sets} sets, "
                          f"{stats.get('successful_runs', 0)}/{stats.get('total_runs', 0)} runs successful)")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Re-analyze ablation study results with detailed statistics"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evals/output_07dec_case",
        help="Base directory containing the results",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output file for analysis JSON (default: <output_dir>/analysis.json)",
    )
    
    args = parser.parse_args()
    
    base_output_dir = Path(args.output_dir)
    
    if not base_output_dir.exists():
        logger.error(f"Output directory not found: {base_output_dir}")
        return 1
    
    # Analyze results
    analysis, detailed_stats = analyze_all_results(base_output_dir)
    
    # Print summary
    print_summary_table(analysis, detailed_stats)
    
    # Save analysis
    output_file = Path(args.output_file) if args.output_file else base_output_dir / "analysis.json"
    with open(output_file, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"\n✓ Analysis saved to: {output_file}")
    
    # Save detailed stats
    detailed_file = base_output_dir / "detailed_stats.json"
    with open(detailed_file, "w") as f:
        json.dump(detailed_stats, f, indent=2)
    logger.info(f"✓ Detailed statistics saved to: {detailed_file}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

