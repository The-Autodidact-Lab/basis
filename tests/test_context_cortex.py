import pytest

from cortex.context_cortex import ContextCortex, cortex


def reset_cortex() -> None:
    cortex._agents.clear()
    cortex._episodes.clear()


def test_parse_and_format_mask():
    c = ContextCortex()

    # Binary-like strings
    assert c.parse_mask("1") == 0b1
    assert c.parse_mask("11") == 0b11
    assert c.parse_mask("101") == 0b101

    # Explicit binary prefix
    assert c.parse_mask("0b101") == 0b101

    # Explicit decimal base
    assert c.parse_mask("10", base=10) == 10

    # Formatting
    assert c.format_mask(0) == "0"
    assert c.format_mask(1) == "1"
    assert c.format_mask(5) == "101"

    with pytest.raises(ValueError):
        c.parse_mask("")


def test_agent_registration_and_mask_update():
    reset_cortex()

    identity = cortex.register_agent("orchestrator", mask=0b10)
    assert identity.agent_id == "orchestrator"
    assert identity.mask == 0b10
    assert cortex.get_agent("orchestrator") is identity

    # Update via binary mask string
    updated = cortex.update_agent_mask("orchestrator", "11")
    assert updated is identity
    assert updated.mask == 0b11

    # Updating a non-existent agent returns None
    assert cortex.update_agent_mask("missing", "1") is None


def test_ingest_episode_and_storage():
    reset_cortex()

    episode = cortex.ingest_episode(
        episode_id="ep_1",
        source_agent_id="orchestrator",
        raw_trace={"logs": ["a", "b"]},
        trace_summary="Orchestrator did something.",
        mask_str="10",
        metadata={"step": 1},
    )

    assert episode.episode_id == "ep_1"
    assert episode.source_agent_id == "orchestrator"
    # "10" (binary) -> 2
    assert episode.access_mask == 0b10
    assert episode.raw_trace == {"logs": ["a", "b"]}
    assert episode.summary == "Orchestrator did something."
    assert episode.metadata["step"] == 1


def test_get_episodes_for_agent_with_and_without_raw():
    reset_cortex()

    # Two episodes with different masks
    cortex.ingest_episode(
        episode_id="ep_a",
        source_agent_id="orchestrator",
        raw_trace="trace_a",
        trace_summary="summary_a",
        mask_str="10",
    )
    cortex.ingest_episode(
        episode_id="ep_b",
        source_agent_id="calendar",
        raw_trace="trace_b",
        trace_summary="summary_b",
        mask_str="01",
    )

    # Agent with mask 0b10 should access only ep_a
    cortex.register_agent("agent_10", mask=0b10)
    visible = cortex.get_episodes_for_agent("agent_10", include_raw=False)
    assert [e.episode_id for e in visible] == ["ep_a"]
    assert visible[0].raw_trace is None
    assert visible[0].summary == "summary_a"

    # Same agent, but include_raw=True should preserve raw_trace
    visible_raw = cortex.get_episodes_for_agent("agent_10", include_raw=True)
    assert [e.episode_id for e in visible_raw] == ["ep_a"]
    assert visible_raw[0].raw_trace == "trace_a"

    # Unknown agent id -> empty list
    assert cortex.get_episodes_for_agent("unknown", include_raw=True) == []


