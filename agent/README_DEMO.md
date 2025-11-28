# Context Cortex Demonstration

This directory contains a demonstration script that shows the Context Cortex in action with the OrchestratorAgent.

## Running the Demo

### Prerequisites

1. Activate the evals virtual environment:
   ```bash
   cd evals
   source .venv/bin/activate
   cd ..
   ```

2. (Optional) Install `google-generativeai` for LLM-powered cortex maintenance:
   ```bash
   # In the evals venv
   pip install google-generativeai
   
   # Or using uv (from root directory)
   uv pip install google-generativeai
   ```

3. (Optional) Set `GEMINI_API_KEY` for LLM-powered cortex maintenance:
   ```bash
   # Add to .env file in root directory:
   GEMINI_API_KEY=your_api_key_here
   
   # Or export in shell:
   export GEMINI_API_KEY=your_api_key_here
   ```

**Note**: The demo will work without `google-generativeai` and `GEMINI_API_KEY`, but will use simple fallback summarization instead of LLM-powered episode summarization and mask proposal.

### Run the Demo

There are two demo scripts:

1. **`agent/demos/demo_cortex_mock.py`** - Uses mock subagents (faster, for testing)
2. **`agent/demos/demo_cortex_real.py`** - Uses REAL ARE apps (Calendar, EmailClientV2)

To run with real ARE apps:
```bash
# From the root directory with evals venv activated
python agent/demos/demo_cortex_real.py
```

Or use the evals venv Python directly:
```bash
evals/.venv/bin/python agent/demos/demo_cortex_real.py
```

To run with mocks (faster):
```bash
evals/.venv/bin/python agent/demos/demo_cortex_mock.py
```

## What the Demo Shows

The demonstration script (`demo_cortex.py`) illustrates:

1. **Context Cortex Setup**: Creating a `ContextCortex` instance
2. **OrchestratorAgent with Cortex**: Creating an orchestrator with cortex enabled and configured
3. **Episode Creation**: How episodes are automatically created after each step
4. **Visibility Control**: How bitmask-based access control determines which agents see which episodes
5. **History Building**: How cortex episodes are merged into agent history before each prompt
6. **Logging**: Detailed logging of all cortex operations including:
   - Agent registration
   - Episode creation and ingestion
   - Visibility matrix
   - History building with cortex episodes

## Example Output

The demo will show:
- Cortex state after each step
- All registered agents with their bitmasks
- All episodes with their access masks
- Visibility matrix showing which agents can see which episodes
- History building showing how cortex episodes are integrated

## Key Features Demonstrated

- **Synchronous Ingestion**: Episodes are ingested synchronously after each step
- **Deduplication**: System logs are filtered out, and duplicate episodes are prevented
- **Mask-based Access**: Agents only see episodes they have access to based on bitmask overlap
- **LLM-powered Summarization**: (If API key is set) Episodes are summarized by a small LLM
- **Automatic Integration**: Cortex is automatically integrated into the agent's history building

## Customization

You can modify `demo_cortex.py` to:
- Use different agent masks
- Add more subagents
- Use real LLM engines instead of mocks
- Test different visibility scenarios
- Add more complex workflows

