#!/usr/bin/env python3
"""
Demonstration script for Context Cortex integration with OrchestratorAgent.

This script shows:
- Setting up a ContextCortex
- Creating an OrchestratorAgent with cortex enabled
- Running a simple task with logging of all cortex operations
- Demonstrating episode creation, visibility, and history building
"""

import logging
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add evals directory to path
_evals_dir = Path(__file__).resolve().parent.parent / "evals"
if str(_evals_dir) not in sys.path:
    sys.path.insert(0, str(_evals_dir))

from agent.context_cortex import ContextCortex
from agent.orchestrator import create_orchestrator
from evals.are.simulation.agents.agent_log import ObservationLog, StepLog, TaskLog, ThoughtLog

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Create a separate logger for cortex operations
cortex_logger = logging.getLogger("cortex")
cortex_logger.setLevel(logging.INFO)


def create_progressive_mock_llm(orchestrator_ref):
    """Create a mock LLM that progresses through the task based on agent state."""
    def mock_llm_engine(prompt, **kwargs):
        """Progressive mock LLM that checks agent state and advances the task."""
        # Check what's been done by looking at recent observations
        has_scheduled = False
        has_sent_email = False
        
        # Get orchestrator from closure
        orchestrator = orchestrator_ref[0] if isinstance(orchestrator_ref, list) else orchestrator_ref
        
        for log in reversed(orchestrator.logs[-20:]):
            if hasattr(log, 'get_type') and log.get_type() == 'observation':
                content = str(log.get_content_for_llm() or "")
                if "calendar" in content.lower() and ("scheduled" in content.lower() or "completed" in content.lower()):
                    has_scheduled = True
                if "email" in content.lower() and ("sent" in content.lower() or "completed" in content.lower()):
                    has_sent_email = True
        
        # Progress through the task
        if not has_scheduled:
            return (
                "Thought: I need to schedule a team meeting for tomorrow at 2pm. "
                "I should delegate this to the calendar agent.\n"
                "Action: {\"action\": \"Calendar__expert_agent\", \"action_input\": "
                "{\"task\": \"Schedule a team meeting for tomorrow at 2pm\"}}<end_action>"
            )
        elif not has_sent_email:
            return (
                "Thought: The meeting has been scheduled. Now I need to send an email to the team. "
                "I should delegate this to the email agent.\n"
                "Action: {\"action\": \"Email__expert_agent\", \"action_input\": "
                "{\"task\": \"Send an email to the team about the meeting scheduled for tomorrow at 2pm\"}}<end_action>"
            )
        else:
            return (
                "Thought: I have successfully scheduled the meeting and sent the email. The task is complete.\n"
                "Action: {\"action\": \"final_answer\", \"action_input\": "
                "{\"answer\": \"Task completed: Team meeting scheduled for tomorrow at 2pm and email sent to the team.\"}}<end_action>"
            )
    return mock_llm_engine


def create_mock_subagent_tool(name: str):
    """Create a mock subagent tool for demonstration."""
    from agent.orchestrator import SubagentTool
    
    def mock_subagent(task: str) -> str:
        logger.info(f"\n  [DELEGATION] Orchestrator delegating to {name} agent")
        logger.info(f"  [DELEGATION] Task: {task}")
        cortex_logger.info(f"  [SUBAGENT {name}] Executing delegated task: {task}")
        result = f"Task '{task}' completed by {name} agent. Meeting scheduled for tomorrow at 2pm." if "calendar" in name.lower() else f"Task '{task}' completed by {name} agent. Email sent to team."
        logger.info(f"  [DELEGATION] {name} agent returned: {result}")
        return result
    
    return SubagentTool(
        name=f"{name}__expert_agent",
        description=f"Delegate a task to the {name} expert agent",
        delegate=mock_subagent,
    )


def log_cortex_state(cortex: ContextCortex, step_name: str):
    """Log the current state of the cortex."""
    cortex_logger.info(f"\n{'='*60}")
    cortex_logger.info(f"CORTEX STATE: {step_name}")
    cortex_logger.info(f"{'='*60}")
    
    # Log all agents
    cortex_logger.info("\nRegistered Agents:")
    for agent_id, identity in cortex._agents.items():
        mask_str = cortex.format_mask(identity.mask)
        cortex_logger.info(f"  - {agent_id}: mask={mask_str} (binary: {bin(identity.mask)})")
    
    # Log all episodes
    cortex_logger.info(f"\nEpisodes ({len(cortex._episodes)} total):")
    for ep_id, episode in cortex._episodes.items():
        mask_str = cortex.format_mask(episode.access_mask)
        cortex_logger.info(f"  - {ep_id}")
        cortex_logger.info(f"    Source: {episode.source_agent_id}")
        cortex_logger.info(f"    Access mask: {mask_str} (binary: {bin(episode.access_mask)})")
        cortex_logger.info(f"    Summary: {episode.summary or '(no summary)'}")
    
    # Log visibility for each agent
    cortex_logger.info("\nVisibility Matrix:")
    for agent_id in cortex._agents.keys():
        visible_eps = cortex.get_episodes_for_agent(agent_id, include_raw=False)
        cortex_logger.info(f"  {agent_id} can see {len(visible_eps)} episodes:")
        for ep in visible_eps:
            cortex_logger.info(f"    - {ep.episode_id} (from {ep.source_agent_id})")


def main():
    """Run the demonstration."""
    logger.info("Starting Context Cortex Demonstration")
    logger.info("=" * 60)
    
    # 1. Create Context Cortex
    logger.info("\n[STEP 1] Creating Context Cortex...")
    cortex = ContextCortex()
    cortex_logger.info("Context Cortex created")
    
    # 2. Create orchestrator with cortex enabled
    logger.info("\n[STEP 2] Creating OrchestratorAgent with cortex...")
    
    # Check if google-generativeai is available
    try:
        import google.generativeai
        genai_available = True
    except ImportError:
        genai_available = False
        logger.warning(
            "google-generativeai not installed. "
            "Install it with: pip install google-generativeai\n"
            "Cortex maintenance will use fallback summarization (simple string truncation)."
        )
    
    # Get API key from environment if available
    api_key = os.getenv("GEMINI_API_KEY")
    if genai_available and not api_key:
        logger.warning(
            "GEMINI_API_KEY not set. "
            "Set it in your .env file or environment to enable LLM-powered cortex maintenance.\n"
            "Cortex maintenance will use fallback summarization."
        )
    elif not genai_available:
        api_key = None  # Don't try to use API key if package isn't installed
    
    # Create mock subagent tools
    calendar_tool = create_mock_subagent_tool("Calendar")
    email_tool = create_mock_subagent_tool("Email")
    
    # Create a closure that will hold the orchestrator reference
    orchestrator_container = {"orchestrator": None}
    
    # Create progressive mock LLM that can access orchestrator state
    def progressive_mock_llm(prompt, **kwargs):
        """Progressive mock LLM that checks orchestrator state."""
        orchestrator = orchestrator_container["orchestrator"]
        if orchestrator is None:
            # Fallback if orchestrator not set yet
            return (
                "Thought: I need to schedule a team meeting for tomorrow at 2pm. "
                "I should delegate this to the calendar agent.\n"
                "Action: {\"action\": \"Calendar__expert_agent\", \"action_input\": "
                "{\"task\": \"Schedule a team meeting for tomorrow at 2pm\"}}<end_action>"
            )
        
        # Check what's been done by looking at recent observations
        has_scheduled = False
        has_sent_email = False
        
        for log in reversed(orchestrator.logs[-20:]):
            if hasattr(log, 'get_type') and log.get_type() == 'observation':
                content = str(log.get_content_for_llm() or "")
                if "calendar" in content.lower() and ("scheduled" in content.lower() or "completed" in content.lower()):
                    has_scheduled = True
                if "email" in content.lower() and ("sent" in content.lower() or "completed" in content.lower()):
                    has_sent_email = True
        
        # Progress through the task
        if not has_scheduled:
            return (
                "Thought: I need to schedule a team meeting for tomorrow at 2pm. "
                "I should delegate this to the calendar agent.\n"
                "Action: {\"action\": \"Calendar__expert_agent\", \"action_input\": "
                "{\"task\": \"Schedule a team meeting for tomorrow at 2pm\"}}<end_action>"
            )
        elif not has_sent_email:
            return (
                "Thought: The meeting has been scheduled. Now I need to send an email to the team. "
                "I should delegate this to the email agent.\n"
                "Action: {\"action\": \"Email__expert_agent\", \"action_input\": "
                "{\"task\": \"Send an email to the team about the meeting scheduled for tomorrow at 2pm\"}}<end_action>"
            )
        else:
            return (
                "Thought: I have successfully scheduled the meeting and sent the email. The task is complete.\n"
                "Action: {\"action\": \"final_answer\", \"action_input\": "
                "{\"answer\": \"Task completed: Team meeting scheduled for tomorrow at 2pm and email sent to the team.\"}}<end_action>"
            )
    
    orchestrator = create_orchestrator(
        llm_engine=progressive_mock_llm,
        subagent_tools=[calendar_tool, email_tool],
        context_cortex=cortex,
        agent_id="orchestrator",
        agent_mask=0b111,  # Can see orchestrator (0b001), calendar (0b010), and email (0b100) episodes
        cortex_maintenance_llm_api_key=api_key,
        other_agents=[
            {"agent_id": "calendar_agent", "mask": 0b010},
            {"agent_id": "email_agent", "mask": 0b100},
        ],
    )
    
    # Store orchestrator reference for the mock LLM
    orchestrator_container["orchestrator"] = orchestrator
    
    cortex_logger.info("OrchestratorAgent created with cortex enabled")
    log_cortex_state(cortex, "After Orchestrator Creation")
    
    # 3. Initialize and run a task
    logger.info("\n[STEP 3] Initializing orchestrator and setting task...")
    orchestrator.initialize()
    
    task = "Schedule a team meeting for tomorrow at 2pm and send an email to the team"
    logger.info(f"Task: {task}")
    
    orchestrator.append_agent_log(
        TaskLog(
            content=task,
            timestamp=orchestrator.make_timestamp(),
            agent_id=orchestrator.agent_id,
        )
    )
    
    # 4. Run the full agent loop
    logger.info("\n[STEP 4] Running full agent loop...")
    logger.info("=" * 60)
    
    max_iterations = 5
    iteration = 0
    
    # Custom termination check
    def should_continue():
        # Check if we've called final_answer
        for log in reversed(orchestrator.logs):
            if hasattr(log, 'get_type') and log.get_type() == 'observation':
                if 'final_answer' in str(log.get_content_for_llm() or '').lower():
                    return False
        return iteration < max_iterations
    
    while should_continue():
        iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}")
        logger.info(f"{'='*60}")
        
        # Log current state before step
        cortex_logger.info(f"\n[PRE-STEP {iteration}] Current cortex state:")
        log_cortex_state(cortex, f"Before Step {iteration}")
        
        # Execute one step of the ReAct loop
        logger.info(f"\n[AGENT STEP {iteration}] Executing orchestrator.step()...")
        try:
            orchestrator.step()
            cortex_logger.info(f"Step {iteration} completed successfully")
            
            # Manually call conditional_post_steps since step() doesn't call them
            # (they're only called in execute_agent_loop())
            if orchestrator.conditional_post_steps:
                for conditional_step in orchestrator.conditional_post_steps:
                    if conditional_step.condition is None or conditional_step.condition(orchestrator):
                        conditional_step.function(orchestrator)
            
            cortex_logger.info(f"\n[POST-STEP {iteration}] Cortex ingestion hook executed")
        except Exception as e:
            logger.error(f"Error in step {iteration}: {e}")
            break
        
        # Log the latest logs with more detail
        logger.info(f"\n[LOGS] Latest agent logs from step {iteration}:")
        recent_logs = orchestrator.logs[-8:]  # Last 8 logs to see full step
        for log in recent_logs:
            log_type = log.get_type() if hasattr(log, 'get_type') else type(log).__name__
            content = ""
            if hasattr(log, 'get_content_for_llm'):
                content = log.get_content_for_llm() or ""
            elif hasattr(log, 'content'):
                content = str(log.content)
            
            # Format based on log type
            if log_type == 'llm_output':
                logger.info(f"  [{log_type}] LLM generated output")
                if content:
                    # Extract thought and action if present
                    if "Thought:" in content:
                        thought = content.split("Action:")[0] if "Action:" in content else content
                        logger.info(f"    Thought: {thought[:150]}...")
                    if "Action:" in content:
                        action = content.split("Action:")[1] if "Action:" in content else ""
                        logger.info(f"    Action: {action[:150]}...")
            elif log_type == 'observation':
                logger.info(f"  [{log_type}] Tool/action result: {str(content)[:150]}")
            elif log_type == 'thought':
                logger.info(f"  [{log_type}]: {str(content)[:150]}")
            elif log_type == 'step':
                logger.info(f"  [{log_type}] Iteration {getattr(log, 'iteration', '?')}")
            else:
                logger.info(f"  [{log_type}]: {str(content)[:100]}")
        
        # Show cortex state after this step
        cortex_logger.info(f"\n[POST-STEP {iteration}] Cortex state after ingestion:")
        log_cortex_state(cortex, f"After Step {iteration}")
        
        # Show history with cortex
        history = orchestrator.build_history_from_logs()
        context_episodes = [m for m in history if "CONTEXT EPISODE" in m.get("content", "")]
        cortex_logger.info(f"\n[HISTORY] History now contains {len(history)} messages, {len(context_episodes)} from cortex")
        
        # Check if we should continue
        if not should_continue():
            logger.info("\n[TERMINATION] Agent called final_answer or max iterations reached")
            break
    
    # 5. Show final state
    logger.info("\n" + "=" * 60)
    logger.info("FINAL STATE")
    logger.info("=" * 60)
    
    log_cortex_state(cortex, "Final State")
    
    # Show final history
    final_history = orchestrator.build_history_from_logs()
    logger.info(f"\n[FINAL HISTORY] Total messages: {len(final_history)}")
    context_episodes = [m for m in final_history if "CONTEXT EPISODE" in m.get("content", "")]
    logger.info(f"Context episodes in final history: {len(context_episodes)}")
    
    # Show summary
    logger.info("\n" + "=" * 60)
    logger.info("DEMONSTRATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total iterations: {iteration}")
    logger.info(f"Total episodes created: {len(cortex._episodes)}")
    logger.info(f"Total agents registered: {len(cortex._agents)}")
    logger.info(f"Total logs: {len(orchestrator.logs)}")
    logger.info("=" * 60)
    logger.info("Demonstration completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

