#!/usr/bin/env python3
"""
Script to automatically run gaia2-mini Agent2Agent phase using the multi_agent agent.

This script runs the Agent2Agent evaluation phase from gaia2-run:
- Multi-agent collaboration scenarios (--a2a_app_prop 1.0)
- gaia2-mini subset (160 scenarios across all capabilities)
- multi_agent agent architecture
- Validation split

This matches the "agent2agent" phase from gaia2-run command, which tests
agent-to-agent collaboration where 100% of apps are replaced by agents.

Usage:
    python run_gaia2_mini_a2a_multi_agent.py [options]

Example:
    python run_gaia2_mini_a2a_multi_agent.py \
        --model gpt-4 \
        --provider openai \
        --output_dir ./gaia2_mini_a2a_results
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Set up logging for this script
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# logger.info(f"GEMINI_API_KEY: {os.getenv('GEMINI_API_KEY')}")

def main():
    parser = argparse.ArgumentParser(
        description="Run gaia2-mini Agent2Agent phase (multi-agent collaboration) with multi_agent agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with OpenAI
  python run_gaia2_mini_a2a_multi_agent.py --model gpt-4 --provider openai

  # With custom output directory and limit
  python run_gaia2_mini_a2a_multi_agent.py \\
      --model gpt-4 \\
      --provider openai \\
      --output_dir ./my_results \\
      --limit 10

  # With custom endpoint
  python run_gaia2_mini_a2a_multi_agent.py \\
      --model llama-3.1-70b \\
      --provider llama-api \\
      --endpoint http://localhost:8000

  # Rate limit friendly (1 scenario at a time)
  python run_gaia2_mini_a2a_multi_agent.py \\
      --model gpt-4 \\
      --provider openai \\
      --rate-limit-friendly

  # Custom concurrency (e.g., 3 concurrent scenarios)
  python run_gaia2_mini_a2a_multi_agent.py \\
      --model gpt-4 \\
      --provider openai \\
      --max_concurrent_scenarios 3

  # Use Gemini for judge instead of OpenAI
  python run_gaia2_mini_a2a_multi_agent.py \\
      --model gpt-4.1 \\
      --provider llama-api \\
      --judge_model gemini-2.5-flash \\
      --judge_provider gemini
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
        default="./gaia2_mini_a2a_multi_agent_results",
        help="Directory to save results (default: ./gaia2_mini_a2a_multi_agent_results)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of scenarios to run (default: all 160 scenarios)",
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=3,
        help="Number of times to run each scenario for variance analysis (default: 3)",
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
        default=2,
        help="Maximum number of concurrent scenarios (default: 2 for rate limit safety). Use --auto-concurrency for CPU-based auto-detect",
    )
    parser.add_argument(
        "--auto-concurrency",
        action="store_true",
        help="Auto-detect concurrency based on CPU count (overrides --max_concurrent_scenarios)",
    )
    parser.add_argument(
        "--rate-limit-friendly",
        action="store_true",
        help="Set conservative concurrency (1 scenario at a time) to respect API rate limits",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for the judge system (default: gpt-4o-mini, use --judge_provider to change provider)",
    )
    parser.add_argument(
        "--judge_provider",
        type=str,
        default="openai",
        help="Provider for the judge model (default: openai, alternatives: gemini/google). Requires API key in environment.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level for benchmark (default: INFO)",
    )
    parser.add_argument(
        "--enable_caching",
        action="store_true",
        help="Enable caching of results",
    )
    parser.add_argument(
        "--executor_type",
        type=str,
        default="process",
        choices=["thread", "process"],
        help="Executor type: 'thread' (more stable, less isolation) or 'process' (more isolation, can fail silently) (default: process)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output including subprocess stdout/stderr",
    )
    parser.add_argument(
        "--save_logs",
        type=str,
        default=None,
        help="Save subprocess stdout/stderr to a log file",
    )

    args = parser.parse_args()

    # Build the command
    cmd = [
        sys.executable,
        "-m",
        "are.simulation.benchmark.cli",
        "run",  # Use 'run' command (not 'gaia2-run' which hardcodes agent="default")
        "--hf-dataset",
        "meta-agents-research-environments/gaia2",
        "--hf-split",
        "validation",
        "--hf-config",
        "mini",  # gaia2-mini subset
        "--agent",
        "multi_agent",  # Use multi_agent agent
        # "--a2a_app_prop",
        # "1.0",  # Agent2Agent phase: 100% of apps become agents (multi-agent collaboration)
        "--model",
        args.model,
        "--provider",
        args.provider,
        "--output_dir",
        args.output_dir,
        "--num_runs",
        str(args.num_runs),
        "--scenario_timeout",
        str(args.scenario_timeout),
        "--log-level",
        args.log_level,
    ]

    # Add optional arguments
    if args.endpoint:
        cmd.extend(["--endpoint", args.endpoint])
    if args.limit:
        cmd.extend(["--limit", str(args.limit)])
    
    # Handle concurrency settings
    if args.rate_limit_friendly:
        # Rate limit friendly mode: run 1 scenario at a time
        max_concurrent = 1
        cmd.extend(["--max_concurrent_scenarios", str(max_concurrent)])
    elif args.auto_concurrency:
        # Auto-detect: don't pass the flag (let benchmark CLI use CPU count)
        pass  # Don't add the flag, let it auto-detect
    else:
        # Use the specified value (default is 2)
        cmd.extend(["--max_concurrent_scenarios", str(args.max_concurrent_scenarios)])
    # Always set judge provider (default is huggingface which works with default judge model)
    cmd.extend(["--judge_provider", args.judge_provider])
    cmd.extend(["--judge_model", args.judge_model])
    if args.enable_caching:
        cmd.append("--enable_caching")
    # Add executor type
    cmd.extend(["--executor_type", args.executor_type])

    # Print configuration
    logger.info("=" * 80)
    logger.info("Running gaia2-mini Agent2Agent phase with multi_agent")
    logger.info("=" * 80)
    logger.info(f"Phase: Agent2Agent (Multi-agent collaboration scenarios)")
    logger.info(f"Model: {args.model}")
    logger.info(f"Provider: {args.provider}")
    if args.endpoint:
        logger.info(f"Endpoint: {args.endpoint}")
    logger.info(f"Agent: multi_agent")
    logger.info(f"Agent-to-agent mode: enabled (a2a_app_prop=1.0)")
    logger.info(f"Dataset: meta-agents-research-environments/gaia2")
    logger.info(f"Split: validation")
    logger.info(f"Config: mini (160 scenarios - gaia2-mini subset)")
    if args.limit:
        logger.info(f"Limit: {args.limit} scenarios")
    logger.info(f"Num runs per scenario: {args.num_runs}")
    logger.info(f"Executor type: {args.executor_type}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Judge model: {args.judge_model}")
    logger.info(f"Judge provider: {args.judge_provider}")
    if args.rate_limit_friendly:
        logger.info(f"Max concurrent scenarios: 1 (rate-limit-friendly mode)")
    elif args.auto_concurrency:
        logger.info(f"Max concurrent scenarios: auto-detect (CPU count)")
    else:
        logger.info(f"Max concurrent scenarios: {args.max_concurrent_scenarios}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    logger.info("")

    # Ensure output directory exists
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run the command
    # Note: subprocess.run() inherits the parent's environment by default,
    # so environment variables loaded via load_dotenv() will be available
    try:
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.info("")
        
        # Let subprocess output stream directly to terminal (preserves tqdm and all output)
        return_code = subprocess.call(
            cmd,
            env=os.environ.copy(),
        )
        
        if return_code != 0:
            logger.error("=" * 80)
            logger.error("✗ Benchmark subprocess failed with return code: %d", return_code)
            logger.error("=" * 80)
            return return_code
        
        logger.info("")
        logger.info("=" * 80)
        logger.info("✓ Successfully completed gaia2-mini agent-to-agent run")
        logger.info(f"Results saved to: {Path(args.output_dir).absolute()}")
        logger.info("=" * 80)
        return 0
        
    except subprocess.CalledProcessError as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("✗ Error running benchmark")
        logger.error(f"Return code: {e.returncode}")
        if hasattr(e, 'stdout') and e.stdout:
            logger.error("STDOUT:")
            logger.error(e.stdout)
        if hasattr(e, 'stderr') and e.stderr:
            logger.error("STDERR:")
            logger.error(e.stderr)
        logger.error("=" * 80)
        return e.returncode
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Interrupted by user")
        return 1
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error("✗ Unexpected error: %s", str(e))
        logger.error("=" * 80)
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

