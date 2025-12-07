#!/usr/bin/env python3
"""
Script to run an ablation study on specific scenarios.

This script runs:
- Three scenarios: case_1_premium_bias, cab_quote_only_vs_book, cab_stale_locations
- Two agent types: default_multi_agent (control) and multi_agent (experimental)
- Five pass^k values: pass^1, pass^2, pass^5, pass^10, pass^20

Results are saved to organized directories, then analyzed to calculate pass^k scores
and generate plots.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Now import matplotlib after logger is defined
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available - plots will be skipped")

# Scenarios to test
# These scenarios follow the Meta Agents Research Environments scenario development pattern:
# https://facebookresearch.github.io/meta-agents-research-environments/tutorials/scenario_development.html
# Each scenario is registered using @register_scenario decorator and implements:
# - init_and_populate_apps(): Initialize and populate apps with data
# - build_events_flow(): Define the sequence of events
# - validate(): Validate that the scenario was completed successfully
SCENARIOS = [
    "case_1_premium_bias",
    "cab_quote_only_vs_book",
    "cab_stale_locations",
]

# Agent types to test
AGENTS = {
    "default_multi_agent": "Control (default_multi_agent)",
    "multi_agent": "Experimental (multi_agent)",
}

# Pass^k values to test
PASS_AT_K_VALUES = [1, 2, 5, 10, 20]

# Number of independent sets to run for each k value
NUM_SETS = 20


def run_scenario(
    scenario_id: str,
    agent: str,
    num_runs: int,
    set_number: int,
    model: str,
    provider: str,
    output_dir: Path,
    endpoint: str | None = None,
    judge_model: str = "gpt-4o-mini",
    judge_provider: str = "openai",
    scenario_timeout: int = 300,
    max_concurrent_scenarios: int = 1,
) -> int:
    """
    Run a scenario with specified parameters for a specific set.
    
    Returns the return code from the subprocess.
    """
    # Create output directory for this specific set
    set_output_dir = output_dir / scenario_id / agent / f"pass_at_{num_runs}" / f"set_{set_number}"
    set_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        "-m",
        "are.simulation.main",
        "-s",
        scenario_id,
        "--agent",
        agent,
        "--model",
        model,
        "--provider",
        provider,
        "--output_dir",
        str(set_output_dir),
        "--scenario_kwargs",
        "{}",
        "--multi_scenario_kwargs",
        "[{}]",
        "--kwargs",
        "{}",
        "--multi_kwargs",
        "[{}]",
        "--log-level",
        "INFO",
    ]
    
    # Add optional arguments
    if endpoint:
        cmd.extend(["--endpoint", endpoint])
    
    # Note: main CLI doesn't have num_runs, so we'll need to run it multiple times
    # or use the benchmark CLI. Actually, let's check if we can use benchmark CLI
    # with scenario IDs. Looking at the code, benchmark CLI uses datasets, not scenario IDs.
    # So we'll use main CLI and run it num_runs times, or we can modify to use
    # a different approach. Actually, let me use the approach where we run
    # the scenario num_runs times by calling it multiple times and aggregating.
    
    # For now, let's use a simpler approach: run the scenario once per run
    # and aggregate results. But that's inefficient. Let me check if main CLI
    # supports multiple runs...
    
    # Actually, I think the best approach is to use the benchmark CLI pattern
    # but create a temporary dataset. Or, we can just run the scenario multiple
    # times and aggregate. Let me use a simpler approach: run num_runs times
            # and save all results, then calculate pass^k from the aggregated results.
    
    logger.info(f"Running {scenario_id} with {agent}, num_runs={num_runs}, set={set_number}")
    logger.info(f"Output directory: {set_output_dir}")
    
    # Run the scenario num_runs times
    all_results = []
    for run_num in range(1, num_runs + 1):
        run_specific_dir = set_output_dir / f"run_{run_num}"
        run_specific_dir.mkdir(parents=True, exist_ok=True)
        
        run_cmd = cmd.copy()
        # Update output_dir in the command (it's the last argument before log-level)
        for i, arg in enumerate(run_cmd):
            if arg == "--output_dir" and i + 1 < len(run_cmd):
                run_cmd[i + 1] = str(run_specific_dir)
                break
        # Also add --export flag to ensure traces are exported
        if "--export" not in run_cmd:
            # Insert before --log-level
            log_level_idx = run_cmd.index("--log-level")
            run_cmd.insert(log_level_idx, "--export")
        
        logger.info(f"  Run {run_num}/{num_runs}: Executing scenario...")
        try:
            result = subprocess.run(
                run_cmd,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=scenario_timeout * 2,  # Allow some buffer
            )
            
            if result.returncode != 0:
                logger.warning(f"  Run {run_num} failed with return code {result.returncode}")
                logger.debug(f"  STDOUT: {result.stdout}")
                logger.debug(f"  STDERR: {result.stderr}")
            
            all_results.append({
                "run_num": run_num,
                "return_code": result.returncode,
                "output_dir": str(run_specific_dir),
            })
        except subprocess.TimeoutExpired:
            logger.warning(f"  Run {run_num} timed out")
            all_results.append({
                "run_num": run_num,
                "return_code": -1,
                "output_dir": str(run_specific_dir),
                "timeout": True,
            })
        except Exception as e:
            logger.error(f"  Run {run_num} raised exception: {e}")
            all_results.append({
                "run_num": run_num,
                "return_code": -1,
                "output_dir": str(run_specific_dir),
                "exception": str(e),
            })
    
    # Save run metadata
    metadata_file = set_output_dir / "run_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump({
            "scenario_id": scenario_id,
            "agent": agent,
            "num_runs": num_runs,
            "set_number": set_number,
            "model": model,
            "provider": provider,
            "runs": all_results,
        }, f, indent=2)
    
    # Return 0 if at least one run succeeded
    successful_runs = [r for r in all_results if r.get("return_code") == 0]
    return 0 if successful_runs else 1


def extract_success_from_json_file(json_file: Path) -> Dict[str, Any] | None:
    """
    Extract success information from a single JSON trace file.
    
    Returns a result dict or None if the file doesn't contain validation info.
    """
    try:
        with open(json_file, "r") as f:
            data = json.load(f)
        
        # Check for validation result in the JSON structure
        # The structure varies, but we look for common patterns
        validation_decision = None
        scenario_id = None
        run_number = None
        
        # Try to find validation decision
        if "validation_decision" in data:
            validation_decision = data["validation_decision"]
        elif "validation" in data and "decision" in data["validation"]:
            validation_decision = data["validation"]["decision"]
        elif "result" in data and "validation_decision" in data["result"]:
            validation_decision = data["result"]["validation_decision"]
        
        # Try to find scenario_id
        if "scenario_id" in data:
            scenario_id = data["scenario_id"]
        elif "scenario" in data and "scenario_id" in data["scenario"]:
            scenario_id = data["scenario"]["scenario_id"]
        
        # Try to find run_number
        if "run_number" in data:
            run_number = data["run_number"]
        
        # Determine success from validation_decision
        # "Valid" or "valid" means success, "Invalid" or "invalid" means failure
        success = False
        if validation_decision:
            success = validation_decision.lower() == "valid"
        
        if scenario_id:
            return {
                "task_id": scenario_id,
                "scenario_id": scenario_id,
                "run_number": run_number,
                "score": 1.0 if success else 0.0,
                "success": success,
                "status": "success" if success else "failed",
                "metadata": {
                    "scenario_id": scenario_id,
                    "run_number": run_number,
                    "validation_decision": validation_decision,
                },
            }
    except (json.JSONDecodeError, KeyError) as e:
        logger.debug(f"Failed to parse JSON file {json_file}: {e}")
    
    return None


def extract_run_number_from_path(file_path: Path) -> int | None:
    """Extract run number from a file path like .../run_3/..."""
    parts = file_path.parts
    for part in parts:
        if part.startswith("run_") and part[4:].isdigit():
            return int(part[4:])
    return None


def extract_success_from_output(output_dir: Path) -> List[Dict[str, Any]]:
    """
    Extract success information from output files organized by sets.
    
    Structure: pass_at_k/set_1/run_1/, set_1/run_2/, ..., set_20/run_k/
    """
    results = []
    
    # Look for set directories
    for set_dir in sorted(output_dir.iterdir()):
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
    
    # Fallback: look for output.jsonl in the directory itself (for backward compatibility)
    if not results:
        output_jsonl = output_dir / "output.jsonl"
        if output_jsonl.exists():
            with open(output_jsonl, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            task_id = data.get("task_id")
                            score = data.get("score", 0.0)
                            metadata = data.get("metadata", {})
                            scenario_id = metadata.get("scenario_id", task_id)
                            run_number = metadata.get("run_number")
                            
                            results.append({
                                "task_id": task_id,
                                "scenario_id": scenario_id,
                                "set_number": None,  # No set number in old format
                                "run_number": run_number,
                                "score": score,
                                "success": score > 0.0,
                                "status": metadata.get("status", "unknown"),
                                "metadata": metadata,
                            })
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse JSON line in {output_jsonl}: {e}")
                            continue
    
    return results


def calculate_pass_hat_k(results: List[Dict[str, Any]], k: int) -> float:
    """
    Calculate pass^k from a list of results organized by sets.
    
    For each set, check if all k runs succeeded. Then average across sets.
    This gives the probability that all k runs succeed.
    
    Pass^k = (number of sets with all k runs successful) / (total sets)
    
    This measures reliability and consistency - all k trials must be successful
    in each set, and we average across multiple independent sets.
    """
    if not results:
        return 0.0
    
    # Group results by set_number
    set_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for result in results:
        set_number = result.get("set_number")
        if set_number is not None:
            set_results[set_number].append(result)
    
    # If no set numbers (backward compatibility), group by scenario_id
    if not set_results:
        scenario_results: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for result in results:
            scenario_id = result.get("scenario_id") or result.get("task_id", "unknown")
            scenario_results[scenario_id].append(result)
        
        # Sort runs by run_number for each scenario
        for scenario_id in scenario_results:
            runs = scenario_results[scenario_id]
            runs.sort(key=lambda x: x.get("run_number") if x.get("run_number") is not None else 999)
        
        # Calculate pass^k for each scenario (old behavior)
        total_scenarios = len(scenario_results)
        if total_scenarios == 0:
            return 0.0
        
        scenarios_with_all_success = 0
        for scenario_id, runs in scenario_results.items():
            scenario_runs = runs[:k]
            if len(scenario_runs) < k:
                continue
            if all(run.get("success", False) for run in scenario_runs):
                scenarios_with_all_success += 1
        
        return scenarios_with_all_success / total_scenarios if total_scenarios > 0 else 0.0
    
    # Sort runs by run_number within each set
    for set_number in set_results:
        runs = set_results[set_number]
        runs.sort(key=lambda x: x.get("run_number", 999))
    
    # Calculate pass^k for each set (binary: all k succeeded or not)
    sets_with_all_success = 0
    total_sets = len(set_results)
    
    if total_sets == 0:
        return 0.0
    
    for set_number, runs in set_results.items():
        # Take only the first k runs for this set
        set_runs = runs[:k]
        
        # Need at least k runs to calculate pass^k
        if len(set_runs) < k:
            continue
        
        # Check if ALL k runs succeeded
        if all(run.get("success", False) for run in set_runs):
            sets_with_all_success += 1
    
    # Average across sets
    return sets_with_all_success / total_sets if total_sets > 0 else 0.0


def analyze_results(base_output_dir: Path) -> Dict[str, Any]:
    """
    Analyze all results and calculate pass^k scores with detailed statistics.
    """
    logger.info("Analyzing results...")
    
    analysis = {}
    
    for scenario_id in SCENARIOS:
        logger.info(f"\n  Scenario: {scenario_id}")
        analysis[scenario_id] = {}
        for agent, agent_label in AGENTS.items():
            logger.info(f"    Agent: {agent_label}")
            analysis[scenario_id][agent] = {}
            
            agent_dir = base_output_dir / scenario_id / agent
            
            if not agent_dir.exists():
                logger.warning(f"      Directory not found: {agent_dir}")
                continue
            
            # For each pass^k value, we need to get results from the corresponding directory
            # Each pass^k directory contains num_runs individual runs
            for k in PASS_AT_K_VALUES:
                pass_k_dir = agent_dir / f"pass_at_{k}"
                
                if not pass_k_dir.exists():
                    logger.warning(f"      Directory not found: {pass_k_dir}")
                    analysis[scenario_id][agent][f"pass_hat_{k}"] = 0.0
                    continue
                
                # Extract results from all runs in this directory
                # Each run is in a subdirectory run_1, run_2, etc.
                all_results = extract_success_from_output(pass_k_dir)
                
                # Calculate pass^k from these results
                # The results should already be from k runs per set, so we calculate pass^k across sets
                pass_hat_k_score = calculate_pass_hat_k(all_results, k)
                analysis[scenario_id][agent][f"pass_hat_{k}"] = pass_hat_k_score
                
                # Log detailed statistics
                total_runs = len(all_results)
                successful_runs = sum(1 for r in all_results if r.get("success", False))
                # Count sets
                sets = set(r.get("set_number") for r in all_results if r.get("set_number") is not None)
                total_sets = len(sets) if sets else 1
                sets_with_all_success = sum(
                    1 for set_num in sets
                    if all(
                        r.get("success", False)
                        for r in all_results
                        if r.get("set_number") == set_num
                        and r.get("run_number", 999) <= k
                    )
                ) if sets else (1 if all(r.get("success", False) for r in all_results[:k]) else 0)
                
                logger.info(f"      Pass^{k}: {pass_hat_k_score*100:.1f}% "
                          f"({sets_with_all_success}/{total_sets} sets with all {k} runs successful, "
                          f"{successful_runs}/{total_runs} total runs successful)")
    
    return analysis


def create_plots(analysis: Dict[str, Any], output_dir: Path):
    """
    Create plots showing pass^k scores for each scenario and agent.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available - skipping plot generation")
        return
    
    logger.info("Creating plots...")
    
    # Create a figure with subplots for each scenario
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(6 * len(SCENARIOS), 6))
    if len(SCENARIOS) == 1:
        axes = [axes]
    
    for idx, scenario_id in enumerate(SCENARIOS):
        ax = axes[idx]
        
        # Plot data for each agent
        for agent, agent_label in AGENTS.items():
            if scenario_id not in analysis or agent not in analysis[scenario_id]:
                continue
            
            k_values = []
            pass_hat_k_scores = []
            
            for k in PASS_AT_K_VALUES:
                key = f"pass_hat_{k}"
                if key in analysis[scenario_id][agent]:
                    k_values.append(k)
                    pass_hat_k_scores.append(analysis[scenario_id][agent][key] * 100)  # Convert to percentage
            
            if k_values:
                ax.plot(k_values, pass_hat_k_scores, marker='o', label=agent_label, linewidth=2, markersize=8)
        
        ax.set_xlabel("k (number of runs)", fontsize=12)
        ax.set_ylabel("Pass^k (%)", fontsize=12)
        ax.set_title(f"{scenario_id}", fontsize=14, fontweight='bold')
        ax.set_xticks(PASS_AT_K_VALUES)
        ax.set_ylim([0, 105])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    plot_path = output_dir / "pass_hat_k_plots.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")
    
    # Also create a summary table plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare data for table
    table_data = []
    headers = ["Scenario", "Agent"] + [f"Pass^{k}" for k in PASS_AT_K_VALUES]
    
    for scenario_id in SCENARIOS:
        for agent, agent_label in AGENTS.items():
            row = [scenario_id, agent_label]
            if scenario_id in analysis and agent in analysis[scenario_id]:
                for k in PASS_AT_K_VALUES:
                    key = f"pass_hat_{k}"
                    score = analysis[scenario_id][agent].get(key, 0.0) * 100
                    row.append(f"{score:.1f}%")
            else:
                row.extend(["N/A"] * len(PASS_AT_K_VALUES))
            table_data.append(row)
    
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.axis('off')
    ax.set_title("Pass^k Results Summary", fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    table_path = output_dir / "pass_hat_k_table.png"
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved table to {table_path}")


def verify_scenarios_registered() -> bool:
    """
    Verify that all required scenarios are registered in the system.
    
    Returns True if all scenarios are found, False otherwise.
    """
    try:
        from are.simulation.scenarios.utils.registry import registry
        
        # Discover scenarios if not already discovered
        if not registry._scenarios_discovered:
            registry._discover_and_import_scenarios()
        
        all_scenarios = registry.get_all_scenarios()
        missing_scenarios = []
        
        for scenario_id in SCENARIOS:
            if scenario_id not in all_scenarios:
                missing_scenarios.append(scenario_id)
        
        if missing_scenarios:
            logger.error(f"The following scenarios are not registered: {missing_scenarios}")
            logger.info("Available scenarios:")
            for sid in sorted(all_scenarios.keys()):
                logger.info(f"  - {sid}")
            return False
        
        logger.info("All required scenarios are registered:")
        for scenario_id in SCENARIOS:
            scenario_class = all_scenarios[scenario_id]
            logger.info(f"  ✓ {scenario_id} -> {scenario_class.__name__}")
        
        return True
    except Exception as e:
        logger.error(f"Error verifying scenarios: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study on specific scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
The scenarios used in this study follow the Meta Agents Research Environments scenario 
development pattern. Each scenario is registered using @register_scenario decorator and 
implements the required methods:
- init_and_populate_apps(): Initialize and populate apps with data
- build_events_flow(): Define the sequence of events  
- validate(): Validate that the scenario was completed successfully

For more information, see:
https://facebookresearch.github.io/meta-agents-research-environments/tutorials/scenario_development.html
        """,
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for inference (e.g., 'gpt-4', 'llama-3.1-70b')",
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=True,
        help="Model provider (e.g., 'openai', 'llama-api', 'huggingface')",
    )
    
    # Optional arguments
    parser.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="Custom endpoint URL for model API",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ablation_study_results",
        help="Base directory to save results (default: ./ablation_study_results)",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for the judge system (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--judge_provider",
        type=str,
        default="openai",
        help="Provider for the judge model (default: openai)",
    )
    parser.add_argument(
        "--scenario_timeout",
        type=int,
        default=300,
        help="Timeout for each scenario in seconds (default: 300)",
    )
    parser.add_argument(
        "--max_concurrent_scenarios",
        type=int,
        default=1,
        help="Maximum number of concurrent scenarios (default: 1)",
    )
    parser.add_argument(
        "--skip_runs",
        action="store_true",
        help="Skip running scenarios and only analyze existing results",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Skip generating plots",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("Ablation Study")
    logger.info("=" * 80)
    
    # Verify scenarios are registered before proceeding
    logger.info("Verifying scenarios are registered...")
    if not verify_scenarios_registered():
        logger.error("Failed to verify scenarios. Please ensure all scenarios are properly registered.")
        return 1
    
    logger.info("")
    logger.info(f"Scenarios: {', '.join(SCENARIOS)}")
    logger.info(f"Agents: {', '.join(AGENTS.keys())}")
    logger.info(f"Pass^k values: {PASS_AT_K_VALUES}")
    logger.info(f"Number of sets per k: {NUM_SETS}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Provider: {args.provider}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    logger.info("")
    
    # Run scenarios
    if not args.skip_runs:
        logger.info("Running scenarios...")
        total_runs = len(SCENARIOS) * len(AGENTS) * len(PASS_AT_K_VALUES) * NUM_SETS
        current_run = 0
        
        for scenario_id in SCENARIOS:
            for agent in AGENTS.keys():
                for num_runs in PASS_AT_K_VALUES:
                    for set_num in range(1, NUM_SETS + 1):
                        current_run += 1
                        logger.info(f"\n[{current_run}/{total_runs}] Running {scenario_id} with {agent}, pass^{num_runs}, set {set_num}/{NUM_SETS}")
                        
                        return_code = run_scenario(
                            scenario_id=scenario_id,
                            agent=agent,
                            num_runs=num_runs,
                            set_number=set_num,
                            model=args.model,
                            provider=args.provider,
                            output_dir=output_dir,
                            endpoint=args.endpoint,
                            judge_model=args.judge_model,
                            judge_provider=args.judge_provider,
                            scenario_timeout=args.scenario_timeout,
                            max_concurrent_scenarios=args.max_concurrent_scenarios,
                        )
                        
                        if return_code != 0:
                            logger.warning(f"Warning: {scenario_id} with {agent} (pass^{num_runs}, set {set_num}) returned code {return_code}")
        
        logger.info("\n" + "=" * 80)
        logger.info("All scenarios completed")
        logger.info("=" * 80)
    
    # Analyze results
    logger.info("\nAnalyzing results...")
    analysis = analyze_results(output_dir)
    
    # Save analysis to JSON
    analysis_file = output_dir / "analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    logger.info(f"Saved analysis to {analysis_file}")
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("Results Summary")
    logger.info("=" * 80)
    for scenario_id in SCENARIOS:
        logger.info(f"\n{scenario_id}:")
        for agent, agent_label in AGENTS.items():
            if scenario_id in analysis and agent in analysis[scenario_id]:
                logger.info(f"  {agent_label}:")
                for k in PASS_AT_K_VALUES:
                    key = f"pass_hat_{k}"
                    if key in analysis[scenario_id][agent]:
                        score = analysis[scenario_id][agent][key] * 100
                        logger.info(f"    Pass^{k}: {score:.1f}%")
    
    # Create plots
    if not args.skip_plots:
        logger.info("\nGenerating plots...")
        try:
            create_plots(analysis, output_dir)
            logger.info("Plots generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
            logger.info("Continuing without plots...")
    
    logger.info("\n" + "=" * 80)
    logger.info("Ablation study complete!")
    logger.info(f"Results saved to: {output_dir.absolute()}")
    logger.info("=" * 80)


if __name__ == "__main__":
    sys.exit(main())

