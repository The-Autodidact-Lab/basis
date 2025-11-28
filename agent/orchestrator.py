"""
Lightweight orchestrator agent built on top of Meta's ARE `BaseAgent`.

Design goals:
- Reuse the existing ReAct-style `BaseAgent` + `JsonActionExecutor`.
- Expose subagents (e.g. AppAgent.expert_agent) as tools the orchestrator can call.
- Keep configuration simple and local to this repository for now.

This module does **not** wire the orchestrator into ARE's CLI yet; it is meant to be
imported from your own code or scenarios. Promotion to a first-class ARE agent can
reuse the same `OrchestratorAgent` implementation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

# Add evals directory to path so 'are' module resolves correctly
# This allows ARE code (which uses 'from are.simulation...') to work
# when running from the root directory
_evals_dir = Path(__file__).resolve().parent.parent / "evals"
if str(_evals_dir) not in sys.path:
    sys.path.insert(0, str(_evals_dir))

from agent.context_cortex import ContextCortex
from agent.cortex_integration import (
    build_history_with_cortex,
    make_cortex_ingestion_hook,
    make_cortex_maintenance_llm,
)
from evals.are.simulation.agents.default_agent.base_agent import (
    BaseAgent as MetaBaseAgent,
    ConditionalStep,
)
from evals.are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from evals.are.simulation.agents.default_agent.default_tools import FinalAnswerTool
from evals.are.simulation.tools import SystemPrompt, Tool


ROOT_DIR = Path(__file__).resolve().parent
ORCHESTRATOR_PROMPT_PATH = ROOT_DIR / "orchestrator_prompt.txt"
SUBAGENT_PROMPT_PATH = ROOT_DIR / "subagent_prompt.txt"


def _load_prompt(path: Path, default: str) -> str:
    """
    Load a prompt from disk, falling back to `default` if the file is missing or empty.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    stripped = text.strip()
    return stripped or default


DEFAULT_ORCHESTRATOR_PROMPT = _load_prompt(
    ORCHESTRATOR_PROMPT_PATH,
    default=(
        "You are an orchestrator agent. You can solve tasks yourself or delegate\n"
        "subtasks to specialized subagents exposed as tools.\n\n"
        "- Think step by step.\n"
        "- When a subagent is clearly a better fit (e.g. Calendar, Email, DB), call\n"
        "  its tool with a clear, self-contained instruction.\n"
        "- After delegating, read the tool's result and decide whether additional\n"
        "  steps or subagent calls are needed.\n"
        "- Always finish by calling the `final_answer` tool with the best overall\n"
        "  answer you can provide.\n"
    ),
)

DEFAULT_SUBAGENT_PROMPT = _load_prompt(
    SUBAGENT_PROMPT_PATH,
    default=(
        "You are an expert subagent for a specific application (e.g. calendar,\n"
        "email, database, or messaging).\n\n"
        "You receive tasks from an orchestrator agent that already did some\n"
        "planning. Your job is to:\n"
        "- Use your application tools to complete the requested work.\n"
        "- Be precise about any changes you make to the app state.\n"
        "- When you are done, call the `final_answer` tool summarizing what you did\n"
        "  and returning any requested result.\n"
    ),
)


class SubagentTool(Tool):
    """
    Minimal Tool wrapper that delegates to a subagent callable.

    The callable should have the signature: (task: str) -> str | object.
    """

    def __init__(
        self,
        name: str,
        description: str,
        delegate: Callable[[str], object],
    ) -> None:
        self.name = name
        self.description = description
        self.inputs = {
            "task": {
                "type": "string",
                "description": "Task description for this subagent to execute.",
            }
        }
        self.output_type = "any"
        super().__init__()
        self._delegate = delegate

    def forward(self, task: str) -> object:
        return self._delegate(task)


def make_app_agent_subagent_tool(app_name: str, app_agent) -> Tool:
    """
    Wrap an ARE `AppAgent` (from `app_agent.py`) as a Tool the orchestrator can call.

    The returned Tool uses the `expert_agent` entry point on the given app_agent.
    """
    description = (
        f"Delegate a task to the {app_name} expert agent via its `expert_agent` tool. "
        "Use this when the task clearly requires that application."
    )
    return SubagentTool(
        name=f"{app_name}__expert_agent",
        description=description,
        delegate=app_agent.expert_agent,
    )


class OrchestratorAgent(MetaBaseAgent):
    """
    Orchestrator built on top of Meta's ARE `BaseAgent`.

    Parameters
    ----------
    llm_engine:
        Callable implementing the ARE LLM engine interface:
        llm_engine(prompt: list[dict], **kwargs) -> str | (str, metadata).
    subagent_tools:
        Mapping from tool name to Tool, typically created via `make_app_agent_subagent_tool`.
    system_prompt:
        Optional override for the orchestrator system prompt. If omitted, a default
        multi-tool orchestration prompt is used.
    context_cortex:
        Optional ContextCortex instance for shared episodic memory.
    agent_id:
        Unique identifier for this agent in the cortex (default: "orchestrator").
    agent_mask:
        Bitmask for access control in cortex (default: 0b11).
    cortex_maintenance_llm_api_key:
        Optional API key for gemini-2.0-flash to use for cortex maintenance.
        If None, cortex ingestion will use simple fallback summaries.
    other_agents:
        List of other agents in the system (for cortex maintenance LLM context).
        Each dict should have 'agent_id' and 'mask' keys.
    """

    def __init__(
        self,
        llm_engine: Callable,
        subagent_tools: Optional[Mapping[str, Tool]] = None,
        system_prompt: Optional[str] = None,
        context_cortex: Optional[ContextCortex] = None,
        agent_id: str = "orchestrator",
        agent_mask: int = 0b11,
        cortex_maintenance_llm_api_key: Optional[str] = None,
        other_agents: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        tools: Dict[str, Tool] = {"final_answer": FinalAnswerTool()}
        if subagent_tools:
            tools.update(dict(subagent_tools))

        # Convert SystemPrompt to string for BaseAgent compatibility
        prompt_str = system_prompt or DEFAULT_ORCHESTRATOR_PROMPT
        system_prompts = {
            "system_prompt": prompt_str
        }

        action_executor = JsonActionExecutor(tools=tools)

        # Store cortex-related state
        self.context_cortex = context_cortex
        self.cortex_agent_id = agent_id
        self._last_cortex_ingestion_index = -1

        # Build conditional post-steps (for cortex ingestion)
        conditional_post_steps = []
        if context_cortex is not None:
            # Register agent in cortex
            context_cortex.register_agent(agent_id, agent_mask)

            # Create maintenance LLM if API key provided
            maintenance_llm = None
            if cortex_maintenance_llm_api_key:
                agent_context = {
                    "agent_id": agent_id,
                    "agent_mask": agent_mask,
                    "other_agents": other_agents or [],
                }
                maintenance_llm = make_cortex_maintenance_llm(
                    cortex_maintenance_llm_api_key, agent_context
                )

            # Create ingestion hook
            ingestion_hook = make_cortex_ingestion_hook(
                cortex=context_cortex,
                agent_id=agent_id,
                agent_mask=agent_mask,
                maintenance_llm=maintenance_llm,
                other_agents=other_agents or [],
            )

            conditional_post_steps.append(
                ConditionalStep(
                    condition=None,  # Always run
                    function=ingestion_hook,
                    name="cortex_ingestion",
                )
            )

        super().__init__(
            llm_engine=llm_engine,
            system_prompts=system_prompts,
            tools=tools,
            action_executor=action_executor,
            conditional_post_steps=conditional_post_steps if conditional_post_steps else None,
        )

    def build_history_from_logs(
        self, exclude_log_types: List[str] = []
    ) -> List[Dict[str, Any]]:
        """
        Build history from logs, with cortex integration if enabled.

        If context_cortex is set, this uses build_history_with_cortex() to merge
        relevant episodes from other agents. Otherwise, falls back to parent method.
        """
        if self.context_cortex is None:
            return super().build_history_from_logs(exclude_log_types=exclude_log_types)

        return build_history_with_cortex(
            agent=self,
            cortex=self.context_cortex,
            agent_id=self.cortex_agent_id,
            exclude_log_types=exclude_log_types,
        )


def create_orchestrator(
    llm_engine: Callable,
    subagent_tools: Iterable[Tool] | Mapping[str, Tool],
    system_prompt: Optional[str] = None,
    context_cortex: Optional[ContextCortex] = None,
    agent_id: str = "orchestrator",
    agent_mask: int = 0b11,
    cortex_maintenance_llm_api_key: Optional[str] = None,
    other_agents: Optional[List[Dict[str, Any]]] = None,
) -> OrchestratorAgent:
    """
    Convenience builder for a fully-wired OrchestratorAgent.

    `subagent_tools` can be:
    - A mapping `name -> Tool`, or
    - An iterable of Tool instances, in which case their `name` attributes are used.
    """
    if isinstance(subagent_tools, Mapping):
        tool_map = dict(subagent_tools)
    else:
        tool_map = {tool.name: tool for tool in subagent_tools}

    return OrchestratorAgent(
        llm_engine=llm_engine,
        subagent_tools=tool_map,
        system_prompt=system_prompt,
        context_cortex=context_cortex,
        agent_id=agent_id,
        agent_mask=agent_mask,
        cortex_maintenance_llm_api_key=cortex_maintenance_llm_api_key,
        other_agents=other_agents,
    )


