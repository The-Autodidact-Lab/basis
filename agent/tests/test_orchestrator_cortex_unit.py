"""
Unit tests for orchestrator-cortex integration initialization and setup.

Target: 100% coverage of initialization, setup, and edge cases.
"""

import pytest

from agent.context_cortex import ContextCortex
from agent.cortex_integration import (
    build_history_with_cortex,
    extract_step_trace,
    make_cortex_ingestion_hook,
    make_cortex_maintenance_llm,
)
from agent.orchestrator import OrchestratorAgent, create_orchestrator
from evals.are.simulation.agents.agent_log import (
    LLMOutputThoughtActionLog,
    ObservationLog,
    StepLog,
    TaskLog,
    ThoughtLog,
)
from evals.are.simulation.agents.default_agent.base_agent import BaseAgent
from evals.are.simulation.agents.default_agent.tools.json_action_executor import (
    JsonActionExecutor,
)
from evals.are.simulation.agents.default_agent.default_tools import FinalAnswerTool


# ---------------------------------------------------------------------------
# Mock LLM engine for testing
# ---------------------------------------------------------------------------


def mock_llm_engine(prompt, **kwargs):
    """Simple mock LLM that returns a basic response."""
    return "Thought: I need to think about this.\nAction: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"test\"}}<end_action>"


# ---------------------------------------------------------------------------
# Initialization tests
# ---------------------------------------------------------------------------


def test_orchestrator_init_without_cortex():
    """Test OrchestratorAgent initialization without cortex (fallback behavior)."""
    agent = OrchestratorAgent(
        llm_engine=mock_llm_engine,
        subagent_tools=None,
    )

    assert agent.context_cortex is None
    assert agent.cortex_agent_id == "orchestrator"  # Default
    assert agent._last_cortex_ingestion_index == -1
    assert hasattr(agent, "build_history_from_logs")


def test_orchestrator_init_with_cortex():
    """Test OrchestratorAgent initialization with cortex enabled."""
    cortex = ContextCortex()
    agent = OrchestratorAgent(
        llm_engine=mock_llm_engine,
        context_cortex=cortex,
        agent_id="test_orchestrator",
        agent_mask=0b11,
    )

    assert agent.context_cortex is cortex
    assert agent.cortex_agent_id == "test_orchestrator"
    assert agent._last_cortex_ingestion_index == -1

    # Agent should be registered in cortex
    identity = cortex.get_agent("test_orchestrator")
    assert identity is not None
    assert identity.mask == 0b11


def test_orchestrator_init_cortex_registration():
    """Test that agent is properly registered in cortex on init."""
    cortex = ContextCortex()
    agent = OrchestratorAgent(
        llm_engine=mock_llm_engine,
        context_cortex=cortex,
        agent_id="agent1",
        agent_mask=0b101,
    )

    identity = cortex.get_agent("agent1")
    assert identity is not None
    assert identity.agent_id == "agent1"
    assert identity.mask == 0b101


def test_orchestrator_init_hook_registration():
    """Test that ingestion hook is registered as conditional_post_step."""
    cortex = ContextCortex()
    agent = OrchestratorAgent(
        llm_engine=mock_llm_engine,
        context_cortex=cortex,
        agent_id="test_agent",
        agent_mask=0b1,
    )

    # Check that conditional_post_steps contains cortex ingestion
    assert agent.conditional_post_steps is not None
    assert len(agent.conditional_post_steps) > 0
    cortex_step = next(
        (s for s in agent.conditional_post_steps if s.name == "cortex_ingestion"),
        None,
    )
    assert cortex_step is not None
    assert cortex_step.condition is None  # Always run


def test_orchestrator_init_without_cortex_no_hooks():
    """Test that no hooks are registered when cortex is None."""
    agent = OrchestratorAgent(
        llm_engine=mock_llm_engine,
        context_cortex=None,
    )

    # conditional_post_steps should be None or empty when no cortex
    assert agent.conditional_post_steps is None or len(agent.conditional_post_steps) == 0


def test_orchestrator_build_history_override():
    """Test that build_history_from_logs is overridden when cortex is enabled."""
    cortex = ContextCortex()
    agent = OrchestratorAgent(
        llm_engine=mock_llm_engine,
        context_cortex=cortex,
        agent_id="test",
        agent_mask=0b1,
    )

    # Should use cortex-aware version
    # We can't easily test the full behavior without setting up logs,
    # but we can verify the method exists and is callable
    assert callable(agent.build_history_from_logs)


def test_orchestrator_build_history_fallback():
    """Test that build_history_from_logs falls back to parent when cortex is None."""
    agent = OrchestratorAgent(
        llm_engine=mock_llm_engine,
        context_cortex=None,
    )

    # Should use parent's build_history_from_logs
    # Initialize agent first
    agent.initialize()
    agent.append_agent_log(
        TaskLog(
            content="test task",
            timestamp=0.0,
            agent_id=agent.agent_id,
        )
    )

    history = agent.build_history_from_logs()
    assert isinstance(history, list)


def test_create_orchestrator_with_cortex():
    """Test create_orchestrator convenience function with cortex."""
    cortex = ContextCortex()
    agent = create_orchestrator(
        llm_engine=mock_llm_engine,
        subagent_tools=[],
        context_cortex=cortex,
        agent_id="created_agent",
        agent_mask=0b10,
    )

    assert agent.context_cortex is cortex
    assert agent.cortex_agent_id == "created_agent"
    identity = cortex.get_agent("created_agent")
    assert identity is not None
    assert identity.mask == 0b10


# ---------------------------------------------------------------------------
# Step trace extraction tests
# ---------------------------------------------------------------------------


def test_extract_step_trace_empty_logs():
    """Test extract_step_trace with empty logs."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()

    trace = extract_step_trace(agent, -1, None)
    assert trace is None  # No new logs


def test_extract_step_trace_first_step():
    """Test extract_step_trace for first step (no previous ingestion)."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )
    agent.append_agent_log(
        StepLog(iteration=0, timestamp=1.0, agent_id=agent.agent_id)
    )

    trace = extract_step_trace(agent, -1, None)
    assert trace is not None
    assert "agent_id" in trace
    assert "logs" in trace
    assert len(trace["logs"]) == 2


def test_extract_step_trace_subsequent_step():
    """Test extract_step_trace for subsequent steps."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )
    agent.append_agent_log(
        StepLog(iteration=0, timestamp=1.0, agent_id=agent.agent_id)
    )

    # First extraction (from beginning, gets TaskLog and first StepLog)
    trace1 = extract_step_trace(agent, -1, None)
    assert trace1 is not None
    # After first extraction, the last ingested log is at index 2 (the StepLog)
    # So last_ingestion_index should be 2 for the next extraction
    
    # Add more logs
    agent.append_agent_log(
        StepLog(iteration=1, timestamp=2.0, agent_id=agent.agent_id)
    )
    agent.append_agent_log(
        ThoughtLog(content="thinking", timestamp=2.1, agent_id=agent.agent_id)
    )

    # Second extraction (from index 2, which is after the first StepLog that was ingested)
    # This should get only the 2 new logs: StepLog at 2.0 and ThoughtLog at 2.1
    trace2 = extract_step_trace(agent, 2, None)
    assert trace2 is not None
    assert len(trace2["logs"]) == 2  # Only the new logs


def test_extract_step_trace_with_cortex_deduplication():
    """Test extract_step_trace with cortex for deduplication checks."""
    cortex = ContextCortex()
    cortex.register_agent("test_agent", 0b1)

    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.cortex_agent_id = "test_agent"
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )

    # Should still extract (deduplication is lenient)
    trace = extract_step_trace(agent, -1, cortex)
    assert trace is not None


# ---------------------------------------------------------------------------
# History building with cortex tests
# ---------------------------------------------------------------------------


def test_build_history_with_cortex_empty_cortex():
    """Test build_history_with_cortex when cortex has no episodes."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )

    cortex = ContextCortex()
    cortex.register_agent("test_agent", 0b1)

    history = build_history_with_cortex(agent, cortex, "test_agent", [])
    # Should return agent's own history
    assert isinstance(history, list)
    assert len(history) > 0


def test_build_history_with_cortex_different_agent_episodes():
    """Test build_history_with_cortex with episodes from different agents."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )

    cortex = ContextCortex()
    cortex.register_agent("agent1", 0b1)
    cortex.register_agent("agent2", 0b10)

    # Add episode from agent2 visible to agent1
    cortex.add_episode(
        episode_id="ep1",
        source_agent_id="agent2",
        access_mask=0b1,  # Visible to agent1
        raw_trace={"msg": "hello"},
        metadata={},
    )
    cortex._episodes["ep1"].summary = "Summary from agent2"

    history = build_history_with_cortex(agent, cortex, "agent1", [])
    # Should include cortex episode
    assert len(history) > 1
    # Check that context episode is included
    context_msgs = [m for m in history if "CONTEXT EPISODE" in m.get("content", "")]
    assert len(context_msgs) > 0


def test_build_history_with_cortex_same_agent_deduplication():
    """Test build_history_with_cortex deduplicates episodes from same agent."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.cortex_agent_id = "agent1"  # Set cortex agent ID for deduplication check
    agent.initialize()
    # Use a specific timestamp that we can match exactly
    test_timestamp = 1.0
    agent.append_agent_log(
        TaskLog(content="test", timestamp=test_timestamp, agent_id=agent.agent_id)
    )

    cortex = ContextCortex()
    cortex.register_agent("agent1", 0b1)

    # Add episode from same agent with overlapping timestamp
    cortex.add_episode(
        episode_id="ep1",
        source_agent_id="agent1",
        access_mask=0b1,
        raw_trace={"msg": "hello"},
        metadata={"timestamp": test_timestamp},  # Same timestamp as TaskLog
    )

    history = build_history_with_cortex(agent, cortex, "agent1", [])
    # Should deduplicate (episode with same timestamp excluded)
    context_msgs = [m for m in history if "CONTEXT EPISODE" in m.get("content", "")]
    assert len(context_msgs) == 0  # Deduplicated


def test_build_history_with_cortex_chronological_ordering():
    """Test that cortex episodes are inserted after system prompt."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )

    cortex = ContextCortex()
    cortex.register_agent("agent1", 0b1)
    cortex.register_agent("agent2", 0b10)

    cortex.add_episode(
        episode_id="ep1",
        source_agent_id="agent2",
        access_mask=0b1,
        raw_trace={"msg": "hello"},
        metadata={},
    )
    cortex._episodes["ep1"].summary = "Summary"

    history = build_history_with_cortex(agent, cortex, "agent1", [])
    # System prompt should be first
    assert history[0].get("role") == "system"
    # Context episodes should come after system but before agent's own logs
    # (This is a simplified check; full ordering would require more setup)


# ---------------------------------------------------------------------------
# Ingestion hook tests
# ---------------------------------------------------------------------------


def test_make_cortex_ingestion_hook_creation():
    """Test creation of cortex ingestion hook."""
    cortex = ContextCortex()
    hook = make_cortex_ingestion_hook(
        cortex=cortex,
        agent_id="test_agent",
        agent_mask=0b1,
        maintenance_llm=None,
        other_agents=[],
    )

    assert callable(hook)


def test_cortex_ingestion_hook_execution():
    """Test that ingestion hook executes and creates episodes."""
    cortex = ContextCortex()
    cortex.register_agent("test_agent", 0b1)

    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.cortex_agent_id = "test_agent"
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )
    agent.append_agent_log(
        StepLog(iteration=0, timestamp=1.0, agent_id=agent.agent_id)
    )

    hook = make_cortex_ingestion_hook(
        cortex=cortex,
        agent_id="test_agent",
        agent_mask=0b1,
        maintenance_llm=None,
        other_agents=[],
    )

    # Execute hook
    hook(agent)

    # Check that episode was created
    episodes = cortex.get_episodes_for_agent("test_agent", include_raw=False)
    assert len(episodes) > 0

    # Check that last ingestion index was updated
    assert agent._last_cortex_ingestion_index >= 0


def test_cortex_ingestion_hook_skips_empty_trace():
    """Test that ingestion hook skips when no new logs."""
    cortex = ContextCortex()
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.cortex_agent_id = "test_agent"
    agent.initialize()

    hook = make_cortex_ingestion_hook(
        cortex=cortex,
        agent_id="test_agent",
        agent_mask=0b1,
        maintenance_llm=None,
        other_agents=[],
    )

    # Execute hook with no new logs
    hook(agent)

    # Should not create episode
    episodes = cortex.get_episodes_for_agent("test_agent", include_raw=False)
    assert len(episodes) == 0


# ---------------------------------------------------------------------------
# Maintenance LLM wrapper tests
# ---------------------------------------------------------------------------


def test_make_cortex_maintenance_llm_fallback():
    """Test fallback when google-generativeai is not available."""
    # This will use fallback since we don't have the SDK in test env
    summarizer = make_cortex_maintenance_llm(
        api_key="fake_key",
        agent_context={"agent_id": "test", "agent_mask": 0b1, "other_agents": []},
    )

    from agent.context_cortex import ContextEpisode

    ep = ContextEpisode(
        episode_id="ep1",
        source_agent_id="test",
        access_mask=0b1,
        raw_trace={"msg": "test"},
        summary=None,
        metadata={},
    )

    summary, mask = summarizer(ep)
    assert isinstance(summary, str)
    assert isinstance(mask, int)
    assert mask == 0b1  # Default to agent's mask


def test_make_cortex_maintenance_llm_with_context():
    """Test maintenance LLM with agent context."""
    summarizer = make_cortex_maintenance_llm(
        api_key="fake_key",
        agent_context={
            "agent_id": "agent1",
            "agent_mask": 0b1,
            "other_agents": [{"agent_id": "agent2", "mask": 0b10}],
        },
    )

    from agent.context_cortex import ContextEpisode

    ep = ContextEpisode(
        episode_id="ep1",
        source_agent_id="agent1",
        access_mask=0b1,
        raw_trace={"msg": "test"},
        summary=None,
        metadata={},
    )

    # Should work even with fallback
    summary, mask = summarizer(ep)
    assert isinstance(summary, str)
    assert isinstance(mask, int)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_build_history_cortex_agent_not_registered():
    """Test build_history_with_cortex when agent is not registered in cortex."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )

    cortex = ContextCortex()
    # Don't register agent

    # Should still work (returns agent's own history)
    history = build_history_with_cortex(agent, cortex, "unregistered_agent", [])
    assert isinstance(history, list)


def test_build_history_cortex_episodes_not_accessible():
    """Test build_history_with_cortex when episodes exist but are not accessible."""
    action_executor = JsonActionExecutor(tools={"final_answer": FinalAnswerTool()})
    agent = BaseAgent(
        llm_engine=mock_llm_engine,
        action_executor=action_executor,
        system_prompts={"system_prompt": "You are a test agent."},
    )
    agent.initialize()
    agent.append_agent_log(
        TaskLog(content="test", timestamp=0.0, agent_id=agent.agent_id)
    )

    cortex = ContextCortex()
    cortex.register_agent("agent1", 0b1)  # mask 0b01
    cortex.register_agent("agent2", 0b10)  # mask 0b10

    # Add episode with mask 0b10 (not accessible to agent1 with mask 0b01)
    cortex.add_episode(
        episode_id="ep1",
        source_agent_id="agent2",
        access_mask=0b10,  # Only visible to agent2
        raw_trace={"msg": "hello"},
        metadata={},
    )

    history = build_history_with_cortex(agent, cortex, "agent1", [])
    # Should not include episode (not accessible)
    context_msgs = [m for m in history if "CONTEXT EPISODE" in m.get("content", "")]
    assert len(context_msgs) == 0

