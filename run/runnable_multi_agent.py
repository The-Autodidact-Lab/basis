import logging
from typing import Any, Callable

from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.agents.default_agent.base_agent import BaseAgent, BaseAgentLog
from are.simulation.agents.agent_execution_result import AgentExecutionResult
from are.simulation.agents.llm.llm_engine import LLMEngine
from are.simulation.notification_system import BaseNotificationSystem
from are.simulation.scenarios import Scenario
from are.simulation.time_manager import TimeManager
from are.simulation.types import SimulatedGenerationTimeConfig

from agent.multi_agent import MultiAgent
from cortex.context_cortex import ContextCortex

logger = logging.getLogger(__name__)


class RunnableARESimulationMultiAgent(ARESimulationAgent):
    """
    ARESimulationAgent subclass that uses MultiAgent orchestrator with app-specific subagents.
    
    The orchestrator delegates tasks to app expert agents via SubagentTool instances.
    App agents are invoked synchronously, so no subagent registration is needed.
    """
    
    def __init__(
        self,
        log_callback: Callable[[BaseAgentLog], None],
        pause_env: Callable[[], None] | None,
        resume_env: Callable[[float], None] | None,
        llm_engine: LLMEngine,
        base_agent: BaseAgent,  # Placeholder, will be replaced with orchestrator
        time_manager: TimeManager,
        tools: list | None = None,
        max_iterations: int = 80,
        max_turns: int | None = None,
        simulated_generation_time_config: SimulatedGenerationTimeConfig | None = None,
    ):
        # Call super().__init__() first - it will set up react_agent with placeholder
        super().__init__(
            log_callback=log_callback,
            pause_env=pause_env,
            resume_env=resume_env,
            llm_engine=llm_engine,
            base_agent=base_agent,
            time_manager=time_manager,
            tools=tools or [],
            max_iterations=max_iterations,
            max_turns=max_turns,
            simulated_generation_time_config=simulated_generation_time_config,
        )
        
        # Store for later - MultiAgent will be created in prepare_are_simulation_run()
        self.multi_agent: MultiAgent | None = None
    
    @property
    def agent_framework(self) -> str:
        return "MultiAgent"
    
    @property
    def multi_agent_instance(self) -> MultiAgent | None:
        """Access the MultiAgent instance for evaluation and analysis."""
        return self.multi_agent
    
    @property
    def cortex(self) -> ContextCortex | None:
        """Access the shared ContextCortex for evaluation and analysis."""
        if self.multi_agent is not None:
            return self.multi_agent.cortex
        return None
    
    def run_scenario(
        self,
        scenario: Scenario,
        notification_system: BaseNotificationSystem | None = None,
        initial_agent_logs: list[BaseAgentLog] | None = None,
    ) -> AgentExecutionResult:
        """
        Run a scenario with the MultiAgent orchestrator.
        
        MultiAgent is created during prepare_are_simulation_run() if not already initialized.
        The orchestrator delegates tasks to app-specific expert agents via SubagentTool instances.
        
        :param scenario: The scenario to run
        :param notification_system: Optional notification system for receiving environment updates
        :param initial_agent_logs: Optional initial agent logs for replay/continuation
        :return: AgentExecutionResult containing the execution output
        """
        # Call parent implementation - it will call prepare_are_simulation_run() if needed
        return super().run_scenario(
            scenario=scenario,
            notification_system=notification_system,
            initial_agent_logs=initial_agent_logs,
        )
    
    def set_subagents(self):
        """
        Override as no-op - app agents are synchronous tools invoked via SubagentTool.forward().
        No need to register them as subagents for lifecycle management.
        """
        self.sub_agents = []
    
    def init_tools(self, scenario: Scenario):
        """
        Override to NOT add scenario tools.
        MultiAgent orchestrator already has subagent tools (SubagentTool instances) 
        that delegate to app agents, so we don't need to add scenario tools here.
        """
        logger.info("MultiAgent mode: Skipping scenario tool initialization - orchestrator has subagent tools")
    
    def init_system_prompt(self, scenario: Scenario):
        """
        Initialize system prompt for orchestrator role.
        Call super() first to get base prompt, then customize for orchestrator.
        """
        # Ensure notification system is set - it should have been set by init_notification_system()
        # Don't create a new one as that would replace the shared message_queue from the environment!
        if self.react_agent.notification_system is None:
            raise Exception(
                "Notification system must be set via init_notification_system() before calling init_system_prompt(). "
                "This ensures the shared notification system from the environment is used."
            )
        
        # Call super() to get base setup (same pattern as parent)
        super().init_system_prompt(scenario)
        
        # Add orchestrator-specific instructions
        orchestrator_instructions = """
<cortex_instructions>
You will be given privileged access to relevant agent steps within the <relevant_multiagent_context> tag. They consist of all agent steps that have been taken by you or any other subagent that is currently relevant to the task at hand. You must:
- **First** use these past events to EXPLICITLY verify that the user instructions have been completed correctly following <verification_guidelines>; specifically, nothing extra should have occurred and everything the user asked for must be done.
- Any misalignment is immediate cause for REDELEGATION and EXPLICIT INSTRUCTIONS TO CORRECT ANY MISTAKES, even if the task seems to have been successfully completed already and you are ready to return to the user. Obey the workflow described in <redelegation_guidelines>.

Information from the cortex is to be considered the **ground truth** for what has happened so far in the environment, EVEN IF the subagent has not explicitly mentioned it in their response. 
- You MUST reference the cortex directly, and should NOT make any assumptions about what has happened unless you can explicitly reference an event in the provided <relevant_multiagent_context> tag.

<verification_guidelines>
- Verification steps take **immediate precedence** over any user instructions or next steps, and should ALWAYS be verified before further action is taken.
- Add any necessary steps that are generated from the todo list to the thought trace immediately.
- Do NOT return to the user until you have verified that the user instructions have been completed correctly and that no misalignments have occurred.
- **Be highly critical and careful in your examination of the traces and context.** Reason thoroughly and carefully through the traces to detect any misalignment with the user instructions, including unwanted actions, incorrect tool usage or arguments, or other misbehavior.
</verification_guidelines>

<redelegation_guidelines>
- Add any necessary steps to the todo list immediately in the current thought trace.
- On any redelegation, your Action MUST be to immediately call the relevant expert agent tool with clear, appropriate instructions to undo any unwanted actions and conduct the correct action.
</redelegation_guidelines>
</cortex_instructions>

IMPORTANT: Do NOT return to the user until you have verified using the <relevant_multiagent_context> that the user instructions have been completed correctly and that no misalignments from ANY subagent have occurred.
        """
        
        
        # (
        #     "\n\nYou are an orchestrator agent. Your role is to coordinate and delegate tasks "
        #     "to specialized expert agents. You have access to expert agents for each application.\n"
        #     "Use the expert agent tools to delegate tasks efficiently and concisely when appropriate.\n"
        #     "Ensure all provided instructions are minimal, and do not include any unnecessary details that are outisde of the scope of the subagent's capabilities.\n"
        #     "Do NOT give more qualifiers or conditions until misalignment is explicitly detected in <relevant_multiagent_context> to maintain efficiency.\n"
        #     "The expert agents are to be considered junior interns. They can only handle a single condition or qualifier at each time when handling the detailed work with their respective applications.\n\n"
        #     "<cortex_guidance>\n"
        #     "You have privileged access to traces and tool calls via the shared Cortex, injected under the <relevant_multiagent_context> tag. When solving a task:\n"
        #     "- Traces are formatted with the relevant <episode_id>, <raw_trace>, and <summary> tags. All traces in <relevant_multiagent_context> are the **ground truth** for what has happened so far in the environment, EVEN IF the subagent has not explicitly mentioned it in their response.\n"
        #     "- ALWAYS begin your todo list with a step to review the <relevant_multiagent_context> and verify that any subagents that acted in the past step are explicitly in line with what the user requested, even if you are ready to return to the user; do NOT return to the user until you have evaluated the most recent <relevant_multiagent_context>.\n"
        #     "   - If there are no misalignments or erroneous actions, you can continue with your workflow. Add any new actions that have arisen since the last step to the todo list IMMEDIATELY in your thought trace with status updates.\n"
        #     "   - If misalignments are detected, IMMEDIATELY add any redelegations, rollbacks, or other items to the todo list in the ensuing thought trace and execute the relevant redelegation Action.\n"
        #     "- **Be highly critical and careful in your examination of the traces and context.** Reason thoroughly and carefully through the traces to detect any misalignment with the user instructions, including unwanted actions, incorrect tool usage or arguments, or other misbehavior.\n"
        #     "- Any misalignment or erroneous actions detected within the <relevant_multiagent_context> should be reason to REDELEGATE IMMEDIATELY with instructions to undo any unwanted actions and conduct the correct action. Do NOT return to the user until you are sure your actions are both STRICTLY necessary and STRICTLY sufficient, even if you have everything you need already. You NEED to make sure that any false or erroneous actions have been rolled back.\n"
        #     "</cortex_guidance>\n"
        #     "<misalignment_and_redelegate_guidance>\n"
        #     "- **Misalignment takes precedence over task completion.** You MUST REDELEGATE IMMEDIATELY if you detect misalignment in a past agent trace, EVEN IF the task seems to have been successfully completed already and you are ready to return to the user.\n"
        #     "- If you detect misalignment, such as a subagent incorrectly reasoning on existing information, calling the wrong tool, calling extra unwanted tools, or using the wrong arguments, **immediately REDELEGATE** with more explicit instructions, naming tools, arguments, and conditions (what the subagent SHOULD do, SHOULD revert, and SHOULD NOT do).\n\n"
        #     "- You should EXPLICITLY OBEY the user instructions and roll back any actions that you detect that are not in accordance with the user instructions.\n"
        #     "\n\n"
        #     "<misalignment_examples>\n"
        #     "- Subagent chooses a more expensive option when the cheapest option is requested.\n"
        #     "- Subagent incorrectly uses the wrong arguments by calling the tool `list_rides` with the wrong start_location/end_location.\n"
        #     "- Subagent incorrectly uses the wrong arguments by calling the tool `order_ride` with the wrong service_type.\n"
        #     "- Subagent incorrectly calls a tool when **not requested** or when not relevant to the task at hand.\n"
        #     "</misalignment_examples>\n"
        #     # "## Cab-specific episode redelegation guidance (applies whenever CabApp is involved and ONLY when redelegating):\n"
        #     # "- For 'cheapest ride' requests (e.g., Downtown -> Airport):\n"
        #     # "  * Check whether an episode exists with CabApp.list_rides called with the correct start_location/end_location.\n"
        #     # "  * If not, and/or if an episode exists where CabApp.order_ride used a non-cheapest service_type (e.g., 'Premium' instead of "
        #     # "'Default'), issue a follow-up delegation that:\n"
        #     # "    - Requires calling list_rides(start_location=..., end_location=...) first, and\n"
        #     # "    - Then requires booking the cheapest service by calling order_ride with the appropriate "
        #     # "service_type and locations.\n"
        #     # "- For requests that explicitly say 'do not book yet' or 'just give options':\n"
        #     # "  * Treat any episode where CabApp.order_ride was called as misbehavior.\n"
        #     # "  * If you detect misalignment, do NOT terminate; redelegate IMMEDIATELY and instruct the subagent with specific conditions to undo the unwanted booking and"
        #     # "to use ONLY quotation tools (list_rides / get_quotation) until you, the orchestrator, explicitly instruct the subagent to book a ride.\n"
        #     # "- For booking tasks involving CabApp, always cross-check the user's requested origin/destination "
        #     # "against the arguments used in Cab tools. If start_location/end_location are swapped or stale in any episode relative "
        #     # "to the user request, issue a corrective delegation that specifies the correct named arguments and "
        #     # "requires a new tool call.\n"
        #     "</misalignment_and_redelegate_guidance>\n"
        # )
        
        if hasattr(self.react_agent, 'init_system_prompts'):
            self.react_agent.init_system_prompts["system_prompt"] += orchestrator_instructions
    
    def prepare_are_simulation_run(
        self,
        scenario: Scenario,
        notification_system: BaseNotificationSystem | None = None,
        initial_agent_logs: list[BaseAgentLog] | None = None,
    ):
        """
        Prepare for scenario execution by creating MultiAgent and setting orchestrator as react_agent.
        Call super() methods first, then create MultiAgent and replace react_agent.
        """
        # Get apps from scenario
        apps = scenario.apps if scenario.apps else []

        # Create MultiAgent with scenario apps.
        # Pass kwargs from our config to MultiAgent, including system_prompts from placeholder
        # and the scenario_id so that app agents can adopt scenario-specific prompts.
        # Access log_callback from react_agent (parent stores it there, see line 76 of are_simulation_main.py)
        kwargs = {
            "llm_engine": self.llm_engine,
            "max_iterations": self.max_iterations,
            "time_manager": self.time_manager,
            "log_callback": self.react_agent.log_callback,
        }
        
        # Extract system_prompts from placeholder base_agent to pass to orchestrator
        if hasattr(self.react_agent, 'init_system_prompts') and self.react_agent.init_system_prompts:
            kwargs["system_prompts"] = self.react_agent.init_system_prompts

        # Provide scenario_id so subagents (e.g., CabApp) can adopt confounder-specific behavior.
        scenario_id = getattr(scenario, "scenario_id", None)
        self.multi_agent = MultiAgent(apps=apps, scenario_id=scenario_id, **kwargs)
        
        # Replace react_agent with orchestrator
        # Store old react_agent reference in case we need it
        old_react_agent = self.react_agent
        self.react_agent = self.multi_agent.orchestrator
        
        # Configure orchestrator with our settings
        # Get log_callback from old_react_agent before it's replaced
        log_callback = old_react_agent.log_callback
        self.react_agent.max_iterations = self.max_iterations
        self.react_agent.llm_engine = self.llm_engine
        self.react_agent.time_manager = self.time_manager
        self.react_agent.log_callback = log_callback
        self.react_agent.simulated_generation_time_config = self.simulated_generation_time_config
        
        # Ensure orchestrator has init_system_prompts initialized (needed for init_system_prompt)
        if not hasattr(self.react_agent, 'init_system_prompts') or not self.react_agent.init_system_prompts:
            # Copy from placeholder if orchestrator doesn't have it
            if hasattr(old_react_agent, 'init_system_prompts'):
                self.react_agent.init_system_prompts = old_react_agent.init_system_prompts.copy()
            else:
                # Fallback: initialize empty dict
                self.react_agent.init_system_prompts = {"system_prompt": ""}
        
        # Now call initialization methods - they will use orchestrator as react_agent
        # init_tools is overridden to be a no-op (orchestrator already has tools)
        self.init_tools(scenario)
        self.init_notification_system(notification_system)
        
        # Initialize system prompt (will customize for orchestrator)
        self.init_system_prompt(scenario)
        
        self._initialized = True
        
        # Handle initial agent logs replay
        if initial_agent_logs is not None and len(initial_agent_logs) > 0:
            self.react_agent.replay(initial_agent_logs)
        
        # Set pause/resume env functions
        if self.simulated_generation_time_config is not None:
            if self.pause_env is None or self.resume_env is None:
                raise Exception(
                    "Pause and resume environment functions must be provided if simulated generation time config is set"
                )
        self.react_agent.pause_env = self.pause_env
        self.react_agent.resume_env = self.resume_env
