"""
Cortex integration helpers for orchestrator-subagent system.

This module provides:
- LLM wrapper for cortex maintenance (gemini-2.0-flash)
- Step trace extraction with deduplication
- History building with cortex episodes
- Post-step ingestion hooks
"""

from __future__ import annotations

import logging
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add evals directory to path so 'are' module resolves correctly
# This allows ARE code (which uses 'from are.simulation...') to work
# when running from the root directory
_evals_dir = Path(__file__).resolve().parent.parent / "evals"
if str(_evals_dir) not in sys.path:
    sys.path.insert(0, str(_evals_dir))

from agent.context_cortex import ContextCortex, ContextEpisode
from evals.are.simulation.agents.agent_log import BaseAgentLog, StepLog
from evals.are.simulation.agents.default_agent.base_agent import BaseAgent

try:
    from are.simulation.agents.llm.types import MessageRole
except ImportError:
    from evals.are.simulation.agents.llm.types import MessageRole

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini maintenance LLM wrapper
# ---------------------------------------------------------------------------


def make_cortex_maintenance_llm(
    api_key: str,
    agent_context: Dict[str, Any],
) -> Callable[[ContextEpisode], Tuple[str, int]]:
    """
    Create a small-LLM callable for cortex maintenance using gemini-2.0-flash.

    Args:
        api_key: Google GenAI API key
        agent_context: Dict with keys:
            - agent_id: str
            - agent_mask: int
            - other_agents: List[Dict[str, Any]] with 'agent_id' and 'mask' keys

    Returns:
        Callable that takes a ContextEpisode and returns (summary: str, access_mask: int)
    """
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning(
            "google-generativeai not installed, cortex maintenance will use fallback"
        )
        return _make_fallback_summarizer(agent_context)

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    agent_id = agent_context.get("agent_id", "unknown")
    agent_mask = agent_context.get("agent_mask", 0)
    other_agents = agent_context.get("other_agents", [])

    def summarizer_and_mask_proposer(ep: ContextEpisode) -> Tuple[str, int]:
        # Build prompt with deduplication awareness
        other_agents_desc = "\n".join(
            [
                f"- Agent {a['agent_id']} (mask {ContextCortex.format_mask(a['mask'])})"
                for a in other_agents
            ]
        )

        prompt = f"""You are a memory curator for an AI agent system.

Current agent: {agent_id} (mask {ContextCortex.format_mask(agent_mask)})
This agent already has access to its own full trace.

Other agents in the system:
{other_agents_desc if other_agents_desc else "None"}

Here is a new episode from agent {ep.source_agent_id}:
Episode ID: {ep.episode_id}
Raw trace: {ep.raw_trace}

Your task:
1. Write a concise one-sentence summary of this episode that avoids repeating information the current agent already sees in its own logs.
2. Propose a binary mask (e.g., "1", "11", "101") for access control. The mask should grant access only to agents that:
   - Need this information for their tasks
   - Don't already have equivalent context in their own traces

Format your response as:
SUMMARY: <one sentence summary>
MASK: <binary mask string, e.g., "1" or "11">

If the episode is redundant or not useful, use MASK: 0"""

        try:
            response = model.generate_content(prompt)
            text = (response.text or "").strip()

            # Parse response
            summary = ""
            mask_str = "0"

            for line in text.split("\n"):
                if line.startswith("SUMMARY:"):
                    summary = line.replace("SUMMARY:", "").strip()
                elif line.startswith("MASK:"):
                    mask_str = line.replace("MASK:", "").strip()

            if not summary:
                summary = f"Episode from {ep.source_agent_id}"

            access_mask = ContextCortex.parse_mask(mask_str, base=2)
            return summary, access_mask

        except Exception as e:
            logger.warning(f"Cortex maintenance LLM failed: {e}, using fallback")
            return _make_fallback_summarizer(agent_context)(ep)

    return summarizer_and_mask_proposer


def _make_fallback_summarizer(
    agent_context: Dict[str, Any],
) -> Callable[[ContextEpisode], Tuple[str, int]]:
    """Fallback summarizer when LLM is unavailable."""

    def fallback(ep: ContextEpisode) -> Tuple[str, int]:
        summary = f"Episode from {ep.source_agent_id}: {str(ep.raw_trace)[:100]}..."
        # Default mask: same as source agent's mask (if available)
        agent_mask = agent_context.get("agent_mask", 0b1)
        return summary, agent_mask

    return fallback


# ---------------------------------------------------------------------------
# Step trace extraction
# ---------------------------------------------------------------------------


def extract_step_trace(
    agent: BaseAgent,
    last_ingestion_index: int,
    cortex: Optional[ContextCortex] = None,
) -> Optional[Any]:
    """
    Extract step-level trace since last ingestion, with deduplication checks.

    Args:
        agent: The agent whose logs to extract
        last_ingestion_index: Index in agent.logs where last ingestion occurred
        cortex: Optional cortex to check for duplicates

    Returns:
        Dict representation of the trace, or None if duplicate/skip
    """
    current_logs = agent.logs[last_ingestion_index + 1 :]

    if not current_logs:
        logger.debug(f"extract_step_trace: No new logs (last_idx={last_ingestion_index}, total={len(agent.logs)})")
        return None

    # Filter out system prompt logs and other initialization logs that aren't part of the step
    # We only want logs that represent actual agent activity (steps, thoughts, actions, etc.)
    filtered_logs = [
        log for log in current_logs
        if log.get_type() != "system_prompt"  # Filter out system prompt logs
    ]

    if not filtered_logs:
        log_types = [log.get_type() for log in current_logs]
        logger.debug(f"extract_step_trace: All logs filtered out. Log types: {log_types}")
        return None
    
    logger.debug(f"extract_step_trace: Extracted {len(filtered_logs)} logs from {len(current_logs)} total")

    # Build compact trace representation
    trace = {
        "agent_id": getattr(agent, "cortex_agent_id", "unknown"),
        "logs": [
            {
                "type": log.get_type(),
                "content": log.get_content_for_llm() if hasattr(log, "get_content_for_llm") else str(log),
                "timestamp": log.timestamp,
            }
            for log in filtered_logs
        ],
    }

    # Deduplication check against cortex
    if cortex is not None:
        agent_id = getattr(agent, "cortex_agent_id", None)
        if agent_id:
            # Check for similar recent episodes from same agent
            recent_episodes = cortex.get_episodes_for_agent(agent_id, include_raw=False)
            if recent_episodes:
                # Simple heuristic: if we have a very recent episode (< 1 second ago) from same agent, skip
                latest_ep = recent_episodes[-1]
                if latest_ep.source_agent_id == agent_id:
                    # Could add more sophisticated content similarity here
                    # For now, we'll ingest anyway but the LLM can decide via mask=0
                    pass

    return trace


# ---------------------------------------------------------------------------
# History building with cortex
# ---------------------------------------------------------------------------


def build_history_with_cortex(
    agent: BaseAgent,
    cortex: ContextCortex,
    agent_id: str,
    exclude_log_types: List[str] = [],
) -> List[Dict[str, Any]]:
    """
    Build history from agent logs + relevant cortex episodes, with deduplication.

    This replaces build_history_from_logs() when cortex is enabled.
    """
    # First, get agent's own history (using parent method directly to avoid recursion)
    # We need to call BaseAgent.build_history_from_logs directly, not the overridden one
    # Get the actual BaseAgent class from the agent's MRO (Method Resolution Order)
    # This finds the first class in the inheritance chain that has build_history_from_logs
    base_class = None
    for cls in agent.__class__.__mro__:
        if hasattr(cls, 'build_history_from_logs') and cls != agent.__class__:
            base_class = cls
            break
    
    if base_class is None:
        # Fallback: import BaseAgent directly
        from evals.are.simulation.agents.default_agent.base_agent import BaseAgent as BaseAgentClass
        base_class = BaseAgentClass
    
    agent_history = base_class.build_history_from_logs(agent, exclude_log_types=exclude_log_types)

    # Get relevant cortex episodes
    cortex_episodes = cortex.get_episodes_for_agent(agent_id, include_raw=False)

    if not cortex_episodes:
        return agent_history

    # Deduplicate: exclude episodes from same agent that overlap with recent logs
    agent_identity = cortex.get_agent(agent_id)
    if not agent_identity:
        return agent_history

    # Get timestamps from agent logs
    agent_timestamps = set()
    for log in agent.logs:
        if hasattr(log, "timestamp"):
            agent_timestamps.add(log.timestamp)

    # Filter cortex episodes
    filtered_episodes = []
    for ep in cortex_episodes:
        # Skip if from same agent and timestamp overlaps (likely duplicate)
        if ep.source_agent_id == agent_id:
            # Check if episode timestamp is in agent's log timestamps
            # This is a simple heuristic; could be improved
            ep_timestamp = ep.metadata.get("timestamp")
            if ep_timestamp and ep_timestamp in agent_timestamps:
                continue  # Skip duplicate
        filtered_episodes.append(ep)

    if not filtered_episodes:
        return agent_history

    # Format cortex episodes as context messages
    cortex_messages = []
    for ep in filtered_episodes:
        summary = ep.summary or f"Context from {ep.source_agent_id}"
        content = f"[CONTEXT EPISODE from {ep.source_agent_id}]: {summary}"
        cortex_messages.append(
            {
                "role": MessageRole.USER,
                "content": content,
                "attachments": None,
            }
        )

    # Merge: system prompt first, then cortex episodes, then agent's own history
    # Find system prompt index
    system_idx = 0
    for i, msg in enumerate(agent_history):
        if msg.get("role") == MessageRole.SYSTEM:
            system_idx = i + 1
            break

    # Insert cortex messages after system prompt
    merged_history = (
        agent_history[:system_idx]
        + cortex_messages
        + agent_history[system_idx:]
    )

    return merged_history


# ---------------------------------------------------------------------------
# Post-step ingestion hook
# ---------------------------------------------------------------------------


def make_cortex_ingestion_hook(
    cortex: ContextCortex,
    agent_id: str,
    agent_mask: int,
    maintenance_llm: Optional[Callable[[ContextEpisode], Tuple[str, int]]],
    other_agents: List[Dict[str, Any]],
) -> Callable[[BaseAgent], None]:
    """
    Create a post-step hook that ingests step traces into cortex.

    This hook runs synchronously after each step and blocks until ingestion completes.
    """
    episode_counter = 0

    def hook(agent: BaseAgent) -> None:
        nonlocal episode_counter

        # Get last ingestion index (stored on agent)
        last_idx = getattr(agent, "_last_cortex_ingestion_index", -1)

        # Extract step trace
        trace = extract_step_trace(agent, last_idx, cortex)
        if trace is None:
            logger.info(
                f"[CORTEX] No new trace to ingest for {agent_id} "
                f"(last_idx={last_idx}, total_logs={len(agent.logs)}, "
                f"logs_since_last={len(agent.logs) - last_idx - 1})"
            )
            # Log what logs exist for debugging
            if len(agent.logs) > last_idx + 1:
                recent_logs = agent.logs[last_idx + 1:]
                log_types = [log.get_type() if hasattr(log, 'get_type') else type(log).__name__ for log in recent_logs]
                logger.info(f"[CORTEX] Recent log types: {log_types}")
            return
        
        logger.info(f"[CORTEX] Ingesting episode for {agent_id}: {len(trace.get('logs', []))} logs")

        # Generate episode ID
        episode_id = f"{agent_id}_ep_{episode_counter}_{uuid.uuid4().hex[:8]}"
        episode_counter += 1

        # Default initial mask (agent's own mask)
        initial_mask = agent_mask
        summary = None

        # If we have a maintenance LLM, use it to get summary and mask
        if maintenance_llm is not None:
            try:
                # Create temporary episode for LLM
                temp_ep = ContextEpisode(
                    episode_id=episode_id,
                    source_agent_id=agent_id,
                    access_mask=initial_mask,
                    raw_trace=trace,
                    summary=None,
                    metadata={"timestamp": agent.make_timestamp()},
                )
                summary, proposed_mask = maintenance_llm(temp_ep)
                initial_mask = proposed_mask
            except Exception as e:
                logger.warning(f"Cortex maintenance LLM failed: {e}, using defaults")
                summary = f"Episode from {agent_id}"

        # Create a summarizer callable if we have a summary
        def summarizer(ep: ContextEpisode) -> str:
            return summary or f"Episode from {agent_id}"

        # Ingest into cortex
        episode = cortex.after_turn_ingest(
            agent_id=agent_id,
            turn_trace=trace,
            initial_mask=initial_mask,
            make_episode_id=lambda: episode_id,
            llm_summarizer=summarizer if summary else None,
            metadata={"timestamp": agent.make_timestamp()},
        )

        # Update last ingestion index
        agent._last_cortex_ingestion_index = len(agent.logs) - 1

    return hook

