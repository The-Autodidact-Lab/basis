import json
from typing import Dict, Any, Callable, List

class SingleReactAgent:

    def __init__(self, tools: Dict[str, Callable], llm_fn: Callable[[str], str], max_steps: int = 10):
        self.tools = tools 
        self.llm_fn = llm_fn           # function is LLM output
        self.max_steps = max_steps     

    def build_prompt(self, task: str, history: List[str]) -> str:
        tool_descs = "\n".join(f"- {name}: {fn.__doc__ or ''}" for name, fn in self.tools.items())
        history_block = "\n".join(history)

        return f"""
You are a single agent using the ReAct pattern (Reason + Act).

TOOLS AVAILABLE:
{tool_descs}

YOUR TASK:
{task}

AT EACH STEP, OUTPUT:
Thought: <your reasoning>
Action: tool_name[{{"arg1": value1, "arg2": value2}}]

OR, WHEN FINISHED:
Thought: <summary>
Final Answer: <answer>

HISTORY:
{history_block}
"""

    def parse_llm(self, output: str):
        thought = ""
        action = None
        final_answer = None

        lines = [l.strip() for l in output.splitlines() if l.strip()]
        for line in lines:
            if line.startswith("Thought:"):
                thought = line[len("Thought:"):].strip()

            elif line.startswith("Action:"):
                rest = line[len("Action:"):].strip()
                tool_name, _, arg_str = rest.partition("[")
                tool_name = tool_name.strip()
                arg_json = arg_str.rsplit("]", 1)[0]
                args = json.loads(arg_json or "{}")
                action = {"tool": tool_name, "args": args}

            elif line.startswith("Final Answer:"):
                final_answer = line[len("Final Answer:"):].strip()

        return thought, action, final_answer

    ## Run agent loop
    def run(self, task: str) -> Dict[str, Any]:
        history = []
        trace = []

        for step in range(self.max_steps):
            prompt = self.build_prompt(task, history)
            llm_output = self.llm_fn(prompt)

            thought, action, final_answer = self.parse_llm(llm_output)
            step_data = {"step": step, "thought": thought, "raw_output": llm_output}

            ## stop if final answer
            if final_answer:
                step_data["final_answer"] = final_answer
                trace.append(step_data)
                return {"success": True, "trace": trace, "answer": final_answer}

            ## result in failure (success = false) if no LLM action
            if not action:
                step_data["error"] = "No action returned by LLM."
                trace.append(step_data)
                return {"success": False, "trace": trace}

            tool_name = action["tool"]
            args = action["args"]

            if tool_name not in self.tools:
                observation = f"ERROR: tool '{tool_name}' not found."
            else:
                try:
                    observation = self.tools[tool_name](**args)
                except Exception as e:
                    observation = f"ERROR: {e}"

            ## save data and update
            step_data["action"] = action
            step_data["observation"] = observation
            trace.append(step_data)

            history.append(f"Thought: {thought}")
            history.append(f"Action: {tool_name}({args})")
            history.append(f"Observation: {observation}")

        trace.append({"error": "Max steps exceeded"})
        return {"success": False, "trace": trace}


## Fake LLM to test

def fake_llm(prompt: str) -> str:
    """
    This fake LLM only outputs one action: add[{"a": 2, "b": 3}]
    It's just to test your loop.
    """
    return """
Thought: I will add the numbers.
Action: add[{"a": 2, "b": 3}]
"""

## Test (temporary)
if __name__ == "__main__":
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    tools = {"add": add}

    agent = SingleReactAgent(tools=tools, llm_fn=fake_llm)
    result = agent.run("Compute 2 + 3.")
    print(result)
