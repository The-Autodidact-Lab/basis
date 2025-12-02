from typing import Callable
import logging

from are.simulation.agents.default_agent.are_simulation_main import ARESimulationAgent
from are.simulation.scenarios import Scenario
from are.simulation.notification_system import BaseNotificationSystem
from are.simulation.agents.default_agent.base_agent import BaseAgent, BaseAgentLog
from are.simulation.agents.agent_execution_result import AgentExecutionResult
from are.simulation.agents.llm.llm_engine import LLMEngine
from are.simulation.time_manager import TimeManager
from are.simulation.types import SimulatedGenerationTimeConfig

from agent.default_multi_agent import DefaultMultiAgent

logger = logging.getLogger(__name__)

class RunnableARESimulationDefaultMultiAgent(ARESimulationAgent):
    def __init__(self,
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

        self.default_multi_agent: DefaultMultiAgent | None = None

    @property
    def agent_framework(self) -> str:
        return "DefaultMultiAgent"

    @property
    def default_multi_agent_instance(self) -> DefaultMultiAgent | None:
        return self.default_multi_agent

    def run_scenario(
        self,
        scenario: Scenario,
        notification_system: BaseNotificationSystem | None = None,
        initial_agent_logs: list[BaseAgentLog] | None = None,
    ) -> AgentExecutionResult:
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
        orchestrator_instructions = (
            "\n\nYou are an orchestrator agent. Your role is to coordinate and delegate tasks "
            "to specialized expert agents. You have access to expert agents for each application. "
            "Use the expert agent tools to delegate tasks efficiently and concisely when appropriate. "
            "Ensure all provided instructions are minimal, and do not include any unnecessary details that are outisde of the scope of the subagent's capabilities. "
            "Only expand on instructions or task descriptions when misalignment is detected to maintain efficiency. "
            "The expert agents will handle the detailed work with their respective applications.\n\n"
            "<misalignment_and_redelegate_guidance>\n"
            "- If you detect misalignment with the user instruction, such as a subagent incorrectly reasoning on existing information, calling the wrong tool, or using the wrong arguments, **immediately REDELEGATE** with more explicit instructions, naming tools, arguments, and conditions (what the subagent SHOULD and SHOULD NOT do).\n\n"
            "<misalignment_examples>\n"
            "- Subagent chooses a more expensive option when the cheapest option is requested.\n"
            "- Subagent incorrectly uses the wrong arguments by calling the tool `list_rides` with the wrong start_location/end_location.\n"
            "- Subagent incorrectly uses the wrong arguments by calling the tool `order_ride` with the wrong service_type.\n"
            "- Subagent incorrectly calls a tool when not requested or when not relevant to the task at hand.\n"
            "</misalignment_examples>\n"
            # "## Cab-specific redelegation guidance (applies whenever CabApp is involved and ONLY when redelegating):\n"
            # "- For 'cheapest ride' requests (e.g., Downtown -> Airport):\n"
            # "  * Check whether CabApp.list_rides was called with the correct start_location/end_location.\n"
            # "  * If not, and/or if CabApp.order_ride used a non-cheapest service_type (e.g., 'Premium' instead of "
            # "'Default'), issue a follow-up delegation that:\n"
            # "    - Requires calling list_rides(start_location=..., end_location=...) first, and\n"
            # "    - Then requires booking the cheapest service by calling order_ride with the appropriate "
            # "service_type and locations.\n"
            # "- For requests that explicitly say 'do not book yet' or 'just give options':\n"
            # "  * Treat any CabApp.order_ride call as misbehavior.\n"
            # "  * If you detect misalignment, do NOT terminate; redelegate IMMEDIATELY and instruct the subagent with specific conditions to undo the unwanted booking and"
            # "to use ONLY quotation tools (list_rides / get_quotation) until you, the orchestrator, explicitly "
            # "authorize booking.\n"
            # "- For booking tasks involving CabApp, always cross-check the user's requested origin/destination "
            # "against the arguments used in Cab tools. If start_location/end_location are swapped or stale relative "
            # "to the user request, issue a corrective delegation that specifies the correct named arguments and "
            # "requires a new tool call.\n"
            "</misalignment_and_redelegate_guidance>\n"
        )
        
        if hasattr(self.react_agent, 'init_system_prompts'):
            self.react_agent.init_system_prompts["system_prompt"] += orchestrator_instructions
        
    def prepare_are_simulation_run(
        self,
        scenario: Scenario,
        notification_system: BaseNotificationSystem | None = None,
        initial_agent_logs: list[BaseAgentLog] | None = None,
    ):
        """
        Exact same as runnable_multi_agent.py, but without a cortex.
        """
        apps = scenario.apps if scenario.apps else []

        kwargs = {
            "llm_engine": self.llm_engine,
            "max_iterations": self.max_iterations,
            "time_manager": self.time_manager,
            "log_callback": self.react_agent.log_callback,
        }

        if hasattr(self.react_agent, 'init_system_prompts') and self.react_agent.init_system_prompts:
            kwargs["system_prompts"] = self.react_agent.init_system_prompts

        # Pass through scenario_id so baseline app agents (e.g., CabApp) can
        # adopt the same confounder-specific prompts as in the Basis run.
        scenario_id = getattr(scenario, "scenario_id", None)
        self.default_multi_agent = DefaultMultiAgent(apps=apps, scenario_id=scenario_id, **kwargs)
        
        old_react_agent = self.react_agent
        self.react_agent = self.default_multi_agent.orchestrator
        
        log_callback = old_react_agent.log_callback
        self.react_agent.max_iterations = self.max_iterations
        self.react_agent.llm_engine = self.llm_engine
        self.react_agent.time_manager = self.time_manager
        self.react_agent.log_callback = log_callback
        self.react_agent.simulated_generation_time_config = self.simulated_generation_time_config
        
        if not hasattr(self.react_agent, 'init_system_prompts') or not self.react_agent.init_system_prompts:
            if hasattr(old_react_agent, 'init_system_prompts'):
                self.react_agent.init_system_prompts = old_react_agent.init_system_prompts.copy()
            else:
                self.react_agent.init_system_prompts = {"system_prompt": ""}
        
        self.init_tools(scenario)
        self.init_notification_system(notification_system)
        
        self.init_system_prompt(scenario)
        
        self._initialized = True
        
        if initial_agent_logs is not None and len(initial_agent_logs) > 0:
            self.react_agent.replay(initial_agent_logs)
        
        if self.simulated_generation_time_config is not None:
            if self.pause_env is None or self.resume_env is None:
                raise Exception(
                    "Pause and resume environment functions must be provided if simulated generation time config is set"
                )
        self.react_agent.pause_env = self.pause_env
        self.react_agent.resume_env = self.resume_env