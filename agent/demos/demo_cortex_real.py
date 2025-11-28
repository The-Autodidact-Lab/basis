#!/usr/bin/env python3
"""
Demonstration script for Context Cortex integration with OrchestratorAgent using REAL ARE apps.

This script shows:
- Setting up a ContextCortex
- Creating an OrchestratorAgent with cortex enabled
- Using REAL ARE apps (Calendar, EmailClientV2) as subagents
- Running a real agent loop with logging of all cortex operations
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
from agent.orchestrator import create_orchestrator, make_app_agent_subagent_tool
from evals.are.simulation.agents.agent_log import ObservationLog, StepLog, TaskLog, ThoughtLog
from evals.are.simulation.agents.agent_builder import AppAgentBuilder
from evals.are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactAppAgentConfig,
    LLMEngineConfig,
)
from evals.are.simulation.apps.calendar import CalendarApp
from evals.are.simulation.apps.email_client import EmailClientV2
from evals.are.simulation.environment import Environment, EnvironmentConfig, EnvironmentType
from evals.are.simulation.notification_system import VerboseNotificationSystem
from evals.are.simulation.agents.llm.llm_engine_builder import LLMEngineBuilder

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
    logger.info("Starting Context Cortex Demonstration with REAL ARE Apps")
    logger.info("=" * 60)
    
    # 1. Create Context Cortex
    logger.info("\n[STEP 1] Creating Context Cortex...")
    cortex = ContextCortex()
    cortex_logger.info("Context Cortex created")
    
    # 2. Set up environment for ARE apps
    logger.info("\n[STEP 2] Setting up ARE environment...")
    env_config = EnvironmentConfig(
        oracle_mode=False,
        queue_based_loop=False,
        wait_for_user_input_timeout=None,
        dump_dir=None,
        time_increment_in_seconds=1.0,
        exit_when_no_events=False,
    )
    env = Environment(
        environment_type=EnvironmentType.CLI,
        config=env_config,
        notification_system=VerboseNotificationSystem(),
    )
    logger.info("Environment created")
    
    # 3. Create real ARE apps
    logger.info("\n[STEP 3] Creating real ARE apps...")
    calendar_app = CalendarApp()
    email_app = EmailClientV2()
    logger.info(f"Created apps: {calendar_app.app_name()}, {email_app.app_name()}")
    
    # 4. Create AppAgents from the apps
    logger.info("\n[STEP 4] Creating AppAgents from apps...")
    
    # Use a simple LLM engine builder (you can configure this)
    llm_engine_builder = LLMEngineBuilder()
    
    # Create app agent configs with proper LLM engine config
    # Use LLAMA_API_KEY if available, otherwise fall back to GEMINI
    llama_api_key = os.getenv("LLAMA_API_KEY")
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    # Allow model name to be overridden via environment variable
    # Default Llama model format: meta-llama/MODEL-NAME or just MODEL-NAME
    llama_model = os.getenv("LLAMA_MODEL", "openai/gpt-5-mini")
    gemini_model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    
    # Create an explicit format system prompt with very clear instructions
    EXPLICIT_FORMAT_SYSTEM_PROMPT = """You are an expert assistant who can solve tasks using JSON tool calls. You will be given a task to solve as best you can.

CRITICAL FORMAT REQUIREMENTS - YOU MUST FOLLOW THIS EXACT FORMAT OR YOU WILL FAIL:

Your response MUST ALWAYS be in this exact format (no exceptions, no variations):

Thought: [Your reasoning about what action to take in plain text]

Action:
{
  "action": "tool_name",
  "action_input": {
    "parameter1": "value1",
    "parameter2": "value2"
  }
}<end_action>

MANDATORY FORMAT RULES (VIOLATING THESE WILL CAUSE ERRORS):
1. ALWAYS start your response with exactly "Thought:" (with colon, no quotes)
2. ALWAYS follow with "Action:" on a new line (with colon, no quotes)
3. ALWAYS provide valid JSON after "Action:" with exactly "action" and "action_input" keys
4. ALWAYS end the JSON with "<end_action>" immediately after the closing brace (no quotes, no spaces)
5. NEVER generate "Observation:" - it will be provided by the system automatically
6. NEVER use markdown code blocks (no ```json or ```)
7. NEVER add text before "Thought:" or after "<end_action>"
8. NEVER return plain text, conversational responses, or questions - ONLY use the Thought/Action format
9. NEVER ask the user questions in your response - use tools to gather information

CORRECT FORMAT EXAMPLE:
Thought: I need to schedule a team meeting for tomorrow at 2pm, so I will call the add_calendar_event tool with the appropriate parameters.

Action:
{
  "action": "add_calendar_event",
  "action_input": {
    "title": "Team Meeting",
    "start_datetime": "2025-11-29 14:00:00",
    "end_datetime": "2025-11-29 15:00:00"
  }
}<end_action>

INCORRECT FORMATS (DO NOT USE THESE):
- Plain text: "I'll schedule the meeting for you..."
- Questions: "Would you like me to schedule the meeting?"
- Markdown: ```json {...} ```
- Missing Thought: Action: {...}
- Missing Action: Thought: I should do something.
- Text after <end_action>: Action: {...}<end_action> Here's what I did...

AVAILABLE TOOLS:
<<tool_descriptions>>

REMEMBER: Your response format is CRITICAL. If you don't follow the exact Thought/Action format, the system will fail to parse your response and you will get an error. Always use this format, no exceptions."""
    
    if llama_api_key:
        logger.info(f"Using LLAMA_API_KEY for subagents with model: {llama_model}")
        app_agent_config = ARESimulationReactAppAgentConfig(
            agent_name="default_app_agent",
            llm_engine_config=LLMEngineConfig(
                model_name=llama_model,
                provider="llama-api",
            ),
            system_prompt=EXPLICIT_FORMAT_SYSTEM_PROMPT,
        )
    elif gemini_api_key:
        logger.info(f"Using GEMINI_API_KEY for subagents with model: {gemini_model}")
        # LiteLLM uses "gemini" as provider for Google Gemini models
        app_agent_config = ARESimulationReactAppAgentConfig(
            agent_name="default_app_agent",
            llm_engine_config=LLMEngineConfig(
                model_name=gemini_model,
                provider="gemini",
            ),
            system_prompt=EXPLICIT_FORMAT_SYSTEM_PROMPT,
        )
    else:
        raise ValueError(
            "Neither LLAMA_API_KEY nor GEMINI_API_KEY found in environment. "
            "Please set one of them in your .env file or environment."
        )
    
    app_agent_builder = AppAgentBuilder(llm_engine_builder=llm_engine_builder)
    
    # Build AppAgents
    calendar_app_agent = app_agent_builder.build(
        agent_config=app_agent_config,
        app=calendar_app,
        env=env,
    )
    calendar_app_agent.cortex_agent_id = "calendar_agent"
    calendar_app_agent._last_cortex_ingestion_index = -1
    
    email_app_agent = app_agent_builder.build(
        agent_config=app_agent_config,
        app=email_app,
        env=env,
    )
    email_app_agent.cortex_agent_id = "email_agent"
    email_app_agent._last_cortex_ingestion_index = -1
    
    # Register subagents with cortex and add ingestion hooks
    from agent.cortex_integration import make_cortex_ingestion_hook, make_cortex_maintenance_llm
    from evals.are.simulation.agents.default_agent.base_agent import ConditionalStep
    
    # Get GEMINI_API_KEY for cortex maintenance (used for episode summarization)
    cortex_maintenance_api_key = gemini_api_key
    
    # Register calendar agent with cortex
    cortex.register_agent("calendar_agent", 0b010)  # Binary mask for calendar
    if cortex_maintenance_api_key:
        calendar_maintenance_llm = make_cortex_maintenance_llm(
            cortex_maintenance_api_key,
            {
                "agent_id": "calendar_agent",
                "agent_mask": 0b010,
                "other_agents": [
                    {"agent_id": "orchestrator", "mask": 0b111},
                    {"agent_id": "email_agent", "mask": 0b100},
                ],
            },
        )
    else:
        calendar_maintenance_llm = None
    
    calendar_ingestion_hook = make_cortex_ingestion_hook(
        cortex, "calendar_agent", 0b010, calendar_maintenance_llm, []
    )
    if calendar_app_agent.app_agent.conditional_post_steps is None:
        calendar_app_agent.app_agent.conditional_post_steps = []
    calendar_app_agent.app_agent.conditional_post_steps.append(
        ConditionalStep(name="cortex_ingestion", function=calendar_ingestion_hook, condition=None)
    )
    
    # Register email agent with cortex
    cortex.register_agent("email_agent", 0b100)  # Binary mask for email
    if cortex_maintenance_api_key:
        email_maintenance_llm = make_cortex_maintenance_llm(
            cortex_maintenance_api_key,
            {
                "agent_id": "email_agent",
                "agent_mask": 0b100,
                "other_agents": [
                    {"agent_id": "orchestrator", "mask": 0b111},
                    {"agent_id": "calendar_agent", "mask": 0b010},
                ],
            },
        )
    else:
        email_maintenance_llm = None
    
    email_ingestion_hook = make_cortex_ingestion_hook(
        cortex, "email_agent", 0b100, email_maintenance_llm, []
    )
    if email_app_agent.app_agent.conditional_post_steps is None:
        email_app_agent.app_agent.conditional_post_steps = []
    email_app_agent.app_agent.conditional_post_steps.append(
        ConditionalStep(name="cortex_ingestion", function=email_ingestion_hook, condition=None)
    )
    
    logger.info("AppAgents created and registered with cortex")
    
    # 5. Create orchestrator with cortex enabled
    logger.info("\n[STEP 5] Creating OrchestratorAgent with cortex...")
    
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
    
    # Create subagent tools from AppAgents
    calendar_tool = make_app_agent_subagent_tool("Calendar", calendar_app_agent)
    email_tool = make_app_agent_subagent_tool("Email", email_app_agent)
    
    # Create a closure that will hold the orchestrator reference for the LLM
    orchestrator_container = {"orchestrator": None}
    
    # Create a progressive mock LLM that can access orchestrator state
    def orchestrator_llm_engine(prompt, **kwargs):
        """Progressive LLM engine that checks orchestrator state."""
        orchestrator = orchestrator_container["orchestrator"]
        if orchestrator is None:
            # Fallback if orchestrator not set yet
            return (
                "Thought: I need to schedule a team meeting for tomorrow at 2pm. I should delegate this to the calendar agent.\n"
                "Action: {\"action\": \"Calendar__expert_agent\", \"action_input\": {\"task\": \"Schedule a team meeting for tomorrow at 2pm\"}}<end_action>"
            )
        
        # Check what's been done by looking at recent observations
        has_scheduled = False
        has_sent_email = False
        
        for log in reversed(orchestrator.logs[-20:]):
            if hasattr(log, 'get_type') and log.get_type() == 'observation':
                content = str(log.get_content_for_llm() or "")
                # Check for calendar scheduling completion - look for event ID or "scheduled" with date/time
                if ("scheduled" in content.lower() and ("event id" in content.lower() or "2025-11-29" in content.lower())) or \
                   ("calendar" in content.lower() and ("scheduled" in content.lower() or "completed" in content.lower())):
                    has_scheduled = True
                # Check for email completion
                if "email" in content.lower() and ("sent" in content.lower() or "completed" in content.lower()):
                    has_sent_email = True
        
        # Progress through the task
        # The task is just to schedule a meeting, so we call final_answer after scheduling
        if not has_scheduled:
            return (
                "Thought: I need to schedule a team meeting for tomorrow at 2pm. I should delegate this to the calendar agent.\n"
                "Action: {\"action\": \"Calendar__expert_agent\", \"action_input\": {\"task\": \"Schedule a team meeting for tomorrow at 2pm\"}}<end_action>"
            )
        else:
            # Meeting has been scheduled, task is complete
            return (
                "Thought: I have successfully scheduled the team meeting for tomorrow at 2pm. The task is complete.\n"
                "Action: {\"action\": \"final_answer\", \"action_input\": {\"answer\": \"Task completed: Team meeting scheduled for tomorrow (2025-11-29) at 2pm.\"}}<end_action>"
            )
    
    orchestrator = create_orchestrator(
        llm_engine=orchestrator_llm_engine,
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
    
    # Store orchestrator reference for the LLM engine
    orchestrator_container["orchestrator"] = orchestrator
    
    cortex_logger.info("OrchestratorAgent created with cortex enabled")
    log_cortex_state(cortex, "After Orchestrator Creation")
    
    # 6. Initialize and run a task
    logger.info("\n[STEP 6] Initializing orchestrator and setting task...")
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
    
    # 7. Run the full agent loop
    logger.info("\n[STEP 7] Running full agent loop...")
    logger.info("=" * 60)
    
    max_iterations = 10
    iteration = 0
    
    # Custom termination check
    def should_continue():
        # Check if we've called final_answer tool
        for log in reversed(orchestrator.logs[-10:]):
            if hasattr(log, 'get_type'):
                log_type = log.get_type()
                # Check for final_answer tool call
                if log_type == 'tool_call':
                    if hasattr(log, 'tool_name') and log.tool_name == 'final_answer':
                        return False
                # Also check observations for final_answer responses
                elif log_type == 'observation':
                    content = str(log.get_content_for_llm() or "")
                    if 'final_answer' in content.lower() or 'task completed' in content.lower():
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
            logger.error(f"Error in step {iteration}: {e}", exc_info=True)
            break
        
        # Log the latest logs
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
    
    # 8. Show final state
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

