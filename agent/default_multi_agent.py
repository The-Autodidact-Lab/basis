from typing import Any

from are.simulation.agents.default_agent.app_agent import AppAgent
from are.simulation.agents.default_agent.base_agent import BaseAgent
from are.simulation.agents.default_agent.default_tools import FinalAnswerTool
from are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor
from are.simulation.agents.default_agent.prompts import (
    DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT,
)
from are.simulation.agents.default_agent.termination_methods.are_simulation import (
    get_gaia2_termination_step,
    termination_step_are_simulation_final_answer,
)
from are.simulation.apps import App
from are.simulation.apps.agent_user_interface import AgentUserInterface
from are.simulation.tools import Tool
from are.simulation.subagent_tool import SubagentTool
from are.simulation.tool_utils import AppToolAdapter

class DefaultMultiAgent:
    orchestrator: BaseAgent
    app_agents: list[AppAgent]
    subagents: list[Tool]

    def __init__(
        self,
        apps: list[App],
        **kwargs: Any,
    ):
        # prevent uninitalised weird stuff
        self.app_agents = []
        aui_tools_dict = {}

        # create default app agents for each app
        for app in apps:
            # AgUI case
            if isinstance(app, AgentUserInterface):
                app.wait_for_user_response = False
                aui_tools = app.get_tools()
                tools_to_remove = {
                    "AgentUserInterface__get_last_message_from_user",
                    "AgentUserInterface__get_last_message_from_agent",
                    "AgentUserInterface__get_last_unread_messages",
                    "AgentUserInterface__get_all_messages",
                }
                filtered_aui_tools = [
                    tool for tool in aui_tools 
                    if tool.name not in tools_to_remove
                ]
                aui_tools_dict = {
                    tool.name: AppToolAdapter(tool) 
                    for tool in filtered_aui_tools
                }
                continue
            
            # all other apps
            app_tools_dict = {
                tool.name: AppToolAdapter(tool) for tool in app.get_tools()
            }
            app_tools_dict["final_answer"] = FinalAnswerTool()

            app_kwargs = {k: v for k, v in kwargs.items() if k != "system_prompts"}
            
            app_base_agent = BaseAgent(
                tools=app_tools_dict,
                action_executor=JsonActionExecutor(tools=app_tools_dict),
                system_prompts={"system_prompt": str(DEFAULT_ARE_SIMULATION_APP_AGENT_REACT_JSON_SYSTEM_PROMPT)},
                termination_step=termination_step_are_simulation_final_answer(),
                **app_kwargs,
            )
            
            # disable notification system for app agents to prevent duplicate message reads and function miscalls
            app_base_agent.notification_system = None

            app_agent = AppAgent(
                app_agent=app_base_agent,
                tools=app_tools_dict,
                name=app.name,
            )
            self.app_agents.append(app_agent)
        
        # create subagents as tools with cortex integration for each app agent
        self.subagents = {
            f"{app_agent.name}__expert_agent": SubagentTool(
                name=f"{app_agent.name}__expert_agent",
                description=f"Delegate a task to the {app_agent.name} expert agent.",
                delegate=app_agent.expert_agent,
            )
            for app_agent in self.app_agents
        }
        
        orchestrator_tools = {**self.subagents, **aui_tools_dict, "final_answer": FinalAnswerTool()}

        self.orchestrator = BaseAgent(
            tools=orchestrator_tools,
            action_executor=JsonActionExecutor(tools=orchestrator_tools),
            termination_step=get_gaia2_termination_step(),
            **kwargs,
        )

    def run(self, task: str):
        return self.orchestrator.run(task=task)