import os
from typing import Any
from dotenv import load_dotenv

load_dotenv()

from evals.are.simulation.agents.default_agent.app_agent import AppAgent
from evals.are.simulation.agents.default_agent.base_cortex_agent import BaseCortexAgent
from evals.are.simulation.agents.default_agent.default_tools import FinalAnswerTool
from evals.are.simulation.agents.default_agent.tools.json_action_executor import JsonActionExecutor
from evals.are.simulation.apps import App
from evals.are.simulation.tools import Tool
from evals.are.simulation.tool_utils import AppToolAdapter

from agent.orchestrator import OrchestratorAgent
from cortex.context_cortex import ContextCortex
from cortex.cortex_agent import CortexAgent

# subagent tool implementation for delegation
class SubagentTool(Tool):
    def __init__(self, name, description, delegate):
        self.name = name
        self.description = description
        self.inputs = {
            "task": {
                "type": "string",
                "description": "Task description for this subagent to execute.",
            }
        }
        self.output_type = "string"
        super().__init__()
        self._delegate = delegate
    
    def forward(self, task: str):
        return self._delegate(task)


class MultiAgent:
    orchestrator: OrchestratorAgent
    cortex: ContextCortex
    cortex_agent: CortexAgent
    app_agents: list[AppAgent]
    subagents: list[Tool]

    def __init__(
        self,
        apps: list[App],
        **kwargs: Any,
    ):
        # initialise cortex/cortex agent
        self.cortex = ContextCortex()
        if os.getenv("GEMINI_API_KEY") is None:
            raise ValueError("GEMINI_API_KEY is not set")
        self.cortex_agent = CortexAgent(api_key=os.getenv("GEMINI_API_KEY"))
        
        self.app_agents = []
        # create app agents with cortex integration for each app
        for i, app in enumerate(apps):
            app_tools_dict = {
                tool.name: AppToolAdapter(tool) for tool in app.get_tools()
            }
            app_tools_dict["final_answer"] = FinalAnswerTool()
            
            app_base_agent = BaseCortexAgent(
                cortex=self.cortex,
                cortex_agent=self.cortex_agent,
                agent_id=f"{app.name}_agent",
                agent_mask=1 << (i + 1),
                tools=app_tools_dict,
                action_executor=JsonActionExecutor(tools=app_tools_dict),
                **kwargs,
            )

            app_agent = AppAgent(
                app_agent=app_base_agent,
                tools=app_tools_dict,
                name=app.name,
            )
            self.app_agents.append(app_agent)

            # cortex registration
            self.cortex.register_agent(
                agent_id=app_base_agent.agent_id,
                mask=app_base_agent.agent_mask,
            )
        
        # create subagents as tools with cortex integration for each app agent
        self.subagents = {
            f"{app_agent.name}__expert_agent": SubagentTool(
                name=f"{app_agent.name}__expert_agent",
                description=f"Delegate a task to the {app_agent.name} expert agent.",
                delegate=app_agent.expert_agent,
            )
            for app_agent in self.app_agents
        }

        # initialise orchestrator with app tools
        self.orchestrator = OrchestratorAgent(
            cortex=self.cortex,
            cortex_agent=self.cortex_agent,
            agent_id="orchestrator",
            agent_mask=0b1,
            tools=self.subagents,
            action_executor=JsonActionExecutor(tools=self.subagents),
            **kwargs,
        )

        self.cortex.register_agent(
            agent_id=self.orchestrator.agent_id,
            mask=self.orchestrator.agent_mask,
        )
        

    def run(self, task: str):
        return self.orchestrator.run(task=task)

