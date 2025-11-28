"""
Integration tests for orchestrator-cortex multi-agent workflows.

Tests verify context visibility for each agent after each step.
"""

import pytest

from agent.context_cortex import ContextCortex
from agent.orchestrator import OrchestratorAgent, create_orchestrator
from evals.are.simulation.agents.agent_log import (
    LLMOutputThoughtActionLog,
    ObservationLog,
    StepLog,
    TaskLog,
    ThoughtLog,
)


# ---------------------------------------------------------------------------
# Mock LLM engines
# ---------------------------------------------------------------------------


def mock_orchestrator_llm(prompt, **kwargs):
    """Mock LLM for orchestrator that delegates to subagents."""
    return "Thought: I should delegate this.\nAction: {\"action\": \"Calendar__expert_agent\", \"action_input\": {\"task\": \"Schedule meeting\"}}<end_action>"


def mock_subagent_llm(prompt, **kwargs):
    """Mock LLM for subagents."""
    return "Thought: I'll handle this.\nAction: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"Meeting scheduled\"}}<end_action>"


# ---------------------------------------------------------------------------
# Multi-agent workflow tests
# ---------------------------------------------------------------------------


def test_multi_agent_workflow_basic():
    """
    Basic multi-agent workflow:
    - Orchestrator + 2 subagents, all registered with cortex
    - Run a simple delegation workflow
    - Verify episodes are created and visible correctly
    """
    cortex = ContextCortex()

    # Create orchestrator
    orchestrator = create_orchestrator(
        llm_engine=mock_orchestrator_llm,
        subagent_tools=[],
        context_cortex=cortex,
        agent_id="orchestrator",
        agent_mask=0b11,  # Can see orchestrator and subagent episodes
        other_agents=[
            {"agent_id": "calendar_agent", "mask": 0b01},
            {"agent_id": "email_agent", "mask": 0b10},
        ],
    )

    # Initialize and run a step
    orchestrator.initialize()
    orchestrator.append_agent_log(
        TaskLog(
            content="Schedule a meeting",
            timestamp=0.0,
            agent_id=orchestrator.agent_id,
        )
    )

    # Simulate a step
    orchestrator.append_agent_log(
        StepLog(iteration=0, timestamp=1.0, agent_id=orchestrator.agent_id)
    )
    orchestrator.append_agent_log(
        ThoughtLog(
            content="I should delegate to calendar",
            timestamp=1.1,
            agent_id=orchestrator.agent_id,
        )
    )

    # Trigger ingestion hook manually
    if orchestrator.conditional_post_steps:
        for step in orchestrator.conditional_post_steps:
            if step.name == "cortex_ingestion":
                step.function(orchestrator)

    # Verify episode was created
    episodes = cortex.get_episodes_for_agent("orchestrator", include_raw=False)
    assert len(episodes) > 0

    # Verify orchestrator can see its own episode
    orchestrator_episodes = cortex.get_episodes_for_agent(
        "orchestrator", include_raw=False
    )
    assert len(orchestrator_episodes) > 0
    assert orchestrator_episodes[0].source_agent_id == "orchestrator"


def test_multi_agent_mask_based_visibility():
    """
    Test mask-based visibility:
    - Orchestrator (mask 0b11) sees episodes with masks 0b01, 0b10, 0b11
    - Subagent A (mask 0b01) sees episodes with masks 0b01, 0b11
    - Subagent B (mask 0b10) sees episodes with masks 0b10, 0b11
    """
    cortex = ContextCortex()

    # Register agents with different masks
    cortex.register_agent("orchestrator", 0b11)
    cortex.register_agent("subagent_a", 0b01)
    cortex.register_agent("subagent_b", 0b10)

    # Create episodes with different access masks
    cortex.add_episode(
        episode_id="ep1",
        source_agent_id="orchestrator",
        access_mask=0b01,  # Visible to subagent_a and orchestrator
        raw_trace={"msg": "episode 1"},
        metadata={},
    )
    cortex.add_episode(
        episode_id="ep2",
        source_agent_id="orchestrator",
        access_mask=0b10,  # Visible to subagent_b and orchestrator
        raw_trace={"msg": "episode 2"},
        metadata={},
    )
    cortex.add_episode(
        episode_id="ep3",
        source_agent_id="subagent_a",
        access_mask=0b11,  # Visible to all
        raw_trace={"msg": "episode 3"},
        metadata={},
    )

    # Check visibility
    orchestrator_eps = cortex.get_episodes_for_agent("orchestrator", include_raw=False)
    assert len(orchestrator_eps) == 3  # Sees all (mask 0b11)

    subagent_a_eps = cortex.get_episodes_for_agent("subagent_a", include_raw=False)
    assert len(subagent_a_eps) == 2  # Sees ep1 (0b01) and ep3 (0b11)
    assert any(ep.episode_id == "ep1" for ep in subagent_a_eps)
    assert any(ep.episode_id == "ep3" for ep in subagent_a_eps)
    assert not any(ep.episode_id == "ep2" for ep in subagent_a_eps)

    subagent_b_eps = cortex.get_episodes_for_agent("subagent_b", include_raw=False)
    assert len(subagent_b_eps) == 2  # Sees ep2 (0b10) and ep3 (0b11)
    assert any(ep.episode_id == "ep2" for ep in subagent_b_eps)
    assert any(ep.episode_id == "ep3" for ep in subagent_b_eps)
    assert not any(ep.episode_id == "ep1" for ep in subagent_b_eps)


def test_multi_step_workflow_context_visibility():
    """
    Test context visibility after multiple steps:
    - Step 1: Orchestrator creates episode
    - Step 2: Subagent A creates episode
    - Step 3: Orchestrator should see both episodes
    """
    cortex = ContextCortex()
    cortex.register_agent("orchestrator", 0b11)
    cortex.register_agent("subagent_a", 0b01)

    # Step 1: Orchestrator episode
    cortex.add_episode(
        episode_id="orchestrator_step1",
        source_agent_id="orchestrator",
        access_mask=0b11,  # Visible to all
        raw_trace={"step": 1, "action": "delegate"},
        metadata={"timestamp": 1.0},
    )
    cortex._episodes["orchestrator_step1"].summary = "Orchestrator delegated task"

    # Step 2: Subagent A episode
    cortex.add_episode(
        episode_id="subagent_a_step1",
        source_agent_id="subagent_a",
        access_mask=0b11,  # Visible to all
        raw_trace={"step": 1, "action": "completed"},
        metadata={"timestamp": 2.0},
    )
    cortex._episodes["subagent_a_step1"].summary = "Subagent A completed task"

    # Step 3: Check orchestrator visibility
    orchestrator_eps = cortex.get_episodes_for_agent("orchestrator", include_raw=False)
    assert len(orchestrator_eps) == 2
    assert any(ep.episode_id == "orchestrator_step1" for ep in orchestrator_eps)
    assert any(ep.episode_id == "subagent_a_step1" for ep in orchestrator_eps)

    # Check subagent A visibility
    subagent_a_eps = cortex.get_episodes_for_agent("subagent_a", include_raw=False)
    assert len(subagent_a_eps) == 2
    assert any(ep.episode_id == "orchestrator_step1" for ep in subagent_a_eps)
    assert any(ep.episode_id == "subagent_a_step1" for ep in subagent_a_eps)


def test_history_building_with_cortex_episodes():
    """
    Test that build_history_with_cortex includes relevant episodes in history.
    """
    cortex = ContextCortex()
    cortex.register_agent("orchestrator", 0b11)
    cortex.register_agent("subagent_a", 0b01)

    # Create orchestrator agent
    orchestrator = create_orchestrator(
        llm_engine=mock_orchestrator_llm,
        subagent_tools=[],
        context_cortex=cortex,
        agent_id="orchestrator",
        agent_mask=0b11,
    )

    orchestrator.initialize()
    orchestrator.append_agent_log(
        TaskLog(
            content="Test task",
            timestamp=0.0,
            agent_id=orchestrator.agent_id,
        )
    )

    # Add episode from subagent that orchestrator should see
    cortex.add_episode(
        episode_id="subagent_ep",
        source_agent_id="subagent_a",
        access_mask=0b11,  # Visible to orchestrator
        raw_trace={"result": "completed"},
        metadata={},
    )
    cortex._episodes["subagent_ep"].summary = "Subagent completed task"

    # Build history
    history = orchestrator.build_history_from_logs()

    # Should include cortex episode
    assert len(history) > 1
    context_msgs = [m for m in history if "CONTEXT EPISODE" in m.get("content", "")]
    assert len(context_msgs) > 0
    assert any("subagent_a" in msg.get("content", "") for msg in context_msgs)


def test_deduplication_in_history_building():
    """
    Test that build_history_with_cortex deduplicates episodes from same agent.
    """
    cortex = ContextCortex()
    cortex.register_agent("orchestrator", 0b11)

    orchestrator = create_orchestrator(
        llm_engine=mock_orchestrator_llm,
        subagent_tools=[],
        context_cortex=cortex,
        agent_id="orchestrator",
        agent_mask=0b11,
    )

    orchestrator.initialize()
    orchestrator.append_agent_log(
        TaskLog(
            content="Test task",
            timestamp=1.0,  # Specific timestamp
            agent_id=orchestrator.agent_id,
        )
    )

    # Add episode from same agent with same timestamp (should be deduplicated)
    cortex.add_episode(
        episode_id="orchestrator_ep",
        source_agent_id="orchestrator",
        access_mask=0b11,
        raw_trace={"msg": "test"},
        metadata={"timestamp": 1.0},  # Same timestamp
    )

    history = orchestrator.build_history_from_logs()
    # Should not include duplicate episode
    context_msgs = [m for m in history if "CONTEXT EPISODE" in m.get("content", "")]
    assert len(context_msgs) == 0  # Deduplicated


def test_ingestion_after_each_step():
    """
    Test that ingestion happens after each step in a multi-step workflow.
    """
    cortex = ContextCortex()

    orchestrator = create_orchestrator(
        llm_engine=mock_orchestrator_llm,
        subagent_tools=[],
        context_cortex=cortex,
        agent_id="orchestrator",
        agent_mask=0b11,
    )

    orchestrator.initialize()
    orchestrator.append_agent_log(
        TaskLog(
            content="Test task",
            timestamp=0.0,
            agent_id=orchestrator.agent_id,
        )
    )

    # Step 1
    orchestrator.append_agent_log(
        StepLog(iteration=0, timestamp=1.0, agent_id=orchestrator.agent_id)
    )
    orchestrator.append_agent_log(
        ThoughtLog(
            content="Step 1 thought",
            timestamp=1.1,
            agent_id=orchestrator.agent_id,
        )
    )

    # Trigger ingestion
    if orchestrator.conditional_post_steps:
        for step in orchestrator.conditional_post_steps:
            if step.name == "cortex_ingestion":
                step.function(orchestrator)

    episodes_after_step1 = cortex.get_episodes_for_agent(
        "orchestrator", include_raw=False
    )
    assert len(episodes_after_step1) == 1

    # Step 2
    orchestrator.append_agent_log(
        StepLog(iteration=1, timestamp=2.0, agent_id=orchestrator.agent_id)
    )
    orchestrator.append_agent_log(
        ThoughtLog(
            content="Step 2 thought",
            timestamp=2.1,
            agent_id=orchestrator.agent_id,
        )
    )

    # Trigger ingestion again
    if orchestrator.conditional_post_steps:
        for step in orchestrator.conditional_post_steps:
            if step.name == "cortex_ingestion":
                step.function(orchestrator)

    episodes_after_step2 = cortex.get_episodes_for_agent(
        "orchestrator", include_raw=False
    )
    assert len(episodes_after_step2) == 2  # Two episodes now


def test_cross_agent_context_sharing():
    """
    Test that agents can see relevant context from other agents.
    """
    cortex = ContextCortex()
    cortex.register_agent("orchestrator", 0b11)
    cortex.register_agent("calendar_agent", 0b01)
    cortex.register_agent("email_agent", 0b10)

    # Orchestrator creates episode visible to calendar agent
    cortex.add_episode(
        episode_id="orchestrator_delegation",
        source_agent_id="orchestrator",
        access_mask=0b01,  # Visible to calendar_agent
        raw_trace={"delegation": "to calendar"},
        metadata={},
    )
    cortex._episodes["orchestrator_delegation"].summary = "Orchestrator delegated to calendar"

    # Calendar agent should see this
    calendar_eps = cortex.get_episodes_for_agent("calendar_agent", include_raw=False)
    assert len(calendar_eps) == 1
    assert calendar_eps[0].episode_id == "orchestrator_delegation"

    # Email agent should NOT see this (different mask)
    email_eps = cortex.get_episodes_for_agent("email_agent", include_raw=False)
    assert len(email_eps) == 0

    # Orchestrator should see it (mask 0b11 overlaps with 0b01)
    orchestrator_eps = cortex.get_episodes_for_agent("orchestrator", include_raw=False)
    assert len(orchestrator_eps) == 1


def test_agent_sees_own_full_trace():
    """
    Test that each agent always sees its own full trace in build_history_from_logs.
    """
    cortex = ContextCortex()

    orchestrator = create_orchestrator(
        llm_engine=mock_orchestrator_llm,
        subagent_tools=[],
        context_cortex=cortex,
        agent_id="orchestrator",
        agent_mask=0b11,
    )

    orchestrator.initialize()
    orchestrator.append_agent_log(
        TaskLog(
            content="Test task",
            timestamp=0.0,
            agent_id=orchestrator.agent_id,
        )
    )
    orchestrator.append_agent_log(
        StepLog(iteration=0, timestamp=1.0, agent_id=orchestrator.agent_id)
    )
    orchestrator.append_agent_log(
        ThoughtLog(
            content="My thought",
            timestamp=1.1,
            agent_id=orchestrator.agent_id,
        )
    )

    history = orchestrator.build_history_from_logs()

    # Should include orchestrator's own logs
    assert len(history) > 0
    # Task should be in history
    task_msgs = [m for m in history if "TASK" in m.get("content", "")]
    assert len(task_msgs) > 0

