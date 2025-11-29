from __future__ import annotations

from typing import Any, Dict, Optional, Union
import os
from dotenv import load_dotenv

load_dotenv()

from evals.are.simulation.agents.default_agent.base_agent import BaseAgent
from evals.are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from evals.are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder
from evals.are.simulation.agents.are_simulation_agent_config import LLMEngineConfig
from evals.are.simulation.agents.llm.types import MMObservation
from evals.are.simulation.tools import Tool

from cortex.context_cortex import ContextCortex, cortex


class IngestEpisodeTool(Tool):
    """
    Tool used by `CortexAgent` to write a new episode into the shared cortex.

    LLM-visible arguments:
    - trace_summary: concise natural-language summary of the episode.
    - mask_str:      access-control mask as a binary-like string (e.g. "1", "11"),
                     chosen by the CortexAgent itself.

    All structural fields (episode_id, source_agent_id, raw_trace, metadata) are
    injected from the surrounding `CortexAgent` via a pending-context getter.
    """

    def __init__(self, get_pending_context) -> None:
        self.name = "ingest_episode"
        self.description = (
            "Ingest the current trace into the shared Cortex. "
            "Call this exactly once per run with a concise trace summary and "
            "the provided access mask string."
        )
        self.inputs = {
            "trace_summary": {
                "type": "string",
                "description": (
                    "Short natural-language summary (2–5 sentences) of the trace "
                    "you were given."
                ),
            },
            "mask_str": {
                "type": "string",
                "description": (
                    "Access-control bitmask string (e.g. '1', '10', '11') that "
                    "determines which agents can see this episode. You, the "
                    "Cortex agent, must decide this mask based on which agents "
                    "should have access."
                ),
            },
        }
        self.output_type = "string"
        super().__init__()
        self._get_pending_context = get_pending_context

    def forward(self, trace_summary: str, mask_str: str) -> str:
        ctx = self._get_pending_context()
        if not ctx:
            return "No pending episode context found; nothing was ingested."

        episode_id: str = ctx["episode_id"]
        source_agent_id: str = ctx["source_agent_id"]
        raw_trace: Any = ctx["raw_trace"]
        metadata: Optional[Dict[str, Any]] = ctx.get("metadata")

        episode = cortex.ingest_episode(
            episode_id=episode_id,
            source_agent_id=source_agent_id,
            raw_trace=raw_trace,
            trace_summary=trace_summary,
            mask_str=mask_str,
            metadata=metadata,
        )
        return f"Episode {episode.episode_id} ingested into Cortex."


class CortexAgent(BaseAgent):
    """
    Lightweight agent that summarizes traces and ingests them into the Cortex.

    Contract:
    - Caller passes in a structured trace plus episode metadata.
    - The agent summarizes the trace and must call `ingest_episode` exactly once
      with `trace_summary` and `mask_str`.
    - The ingest tool then writes the full episode into the shared `cortex`.
    """

    api_key: str = os.getenv("GEMINI_API_KEY")
    model: str = "gemini-2.0-flash"
    provider: str = "gemini"
    system_prompt: str = """
You are a Cortex agent. You are responsible for maintaining the shared Cortex
memory for all agents.

Given a specific trace:
- Carefully read and understand what happened.
- Produce a concise, high-signal summary (2–5 sentences).
- Propose an access-control mask as a binary string (e.g. "1", "10", "11") that
  indicates which agent groups should see this episode.
- Then call the `ingest_episode` tool EXACTLY ONCE with:
  - `trace_summary`: your summary text.
  - `mask_str`: the mask string you chose.

Do not call any other tools. If you are unsure about the exact mask, choose a
reasonable default and clearly document your choice in the summary.
"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self._pending_ingest: Optional[Dict[str, Any]] = None

        # set up ingest tool
        ingest_tool = IngestEpisodeTool(get_pending_context=self._get_pending_ingest)
        tools: Dict[str, Tool] = {"ingest_episode": ingest_tool}

        engine_config = LLMEngineConfig(
            model_name=self.model,
            provider=self.provider
        )
        llm_engine = LLMEngineBuilder().create_engine(engine_config)

        action_executor = JsonActionExecutor(tools=tools)

        super().__init__(
            llm_engine=llm_engine,
            system_prompts={"system_prompt": self.system_prompt},
            tools=tools,
            action_executor=action_executor,
            max_iterations=2,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_pending_ingest(self) -> Optional[Dict[str, Any]]:
        return self._pending_ingest

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        episode_id: str,
        source_agent_id: str,
        raw_trace: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Summarize a trace and ingest it into the Cortex via the ingest tool.

        The caller is responsible for:
        - Choosing `episode_id` (unique per episode) and `source_agent_id`.
        - Providing `raw_trace` in any serializable / stringifiable form.
        - Choosing an appropriate access-control mask string; the agent will
          encode this as `mask_str` in the ingest tool call.
        """
        self._pending_ingest = {
            "episode_id": episode_id,
            "source_agent_id": source_agent_id,
            "raw_trace": raw_trace,
            "metadata": metadata or {},
        }

        # Build the task for the underlying ReAct loop.
        trace_repr = str(raw_trace)
        task = (
            "You are given an agent trace.\n\n"
            "[TRACE]\n"
            f"{trace_repr}\n\n"
            "Your job:\n"
            "1. Read and understand the trace.\n"
            "2. Produce a concise, high-signal summary (2–5 sentences).\n"
            "3. Decide on an appropriate access-control mask string `mask_str` "
            '(binary like "1", "10", "11") indicating which agent groups should '
            "see this episode.\n"
            "4. Call the `ingest_episode` tool EXACTLY ONCE with:\n"
            "   - trace_summary: your summary\n"
            "   - mask_str: the mask string you chose\n"
            "After calling the tool, you may provide a very short confirmation.\n"
        )

        try:
            result: Union[str, MMObservation, None] = super().run(task, reset=True)
        finally:
            # Clear context to avoid accidental reuse across runs.
            self._pending_ingest = None

        if isinstance(result, MMObservation):
            return result.content
        if result is None:
            return ""
        return str(result)