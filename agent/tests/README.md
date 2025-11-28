# Agent Tests

## Running Tests

**IMPORTANT**: These tests must be run with the `evals/.venv` environment activated, as they depend on ARE (Meta Agents Research Environments) which is installed there.

### Option 1: Use the test runner script (recommended)

```bash
./run_agent_tests.sh
```

### Option 2: Manual activation

```bash
cd evals
source .venv/bin/activate
cd ..
pytest agent/tests/
```

### Option 3: Direct venv Python

```bash
evals/.venv/bin/python -m pytest agent/tests/
```

## Test Suites

- **Unit tests** (`test_orchestrator_cortex_unit.py`): Test initialization, setup, and edge cases (target: 100% coverage)
- **Integration tests** (`test_orchestrator_cortex_integration.py`): Test multi-agent workflows and context visibility
- **Context cortex tests** (`test_context_cortex_*.py`): Test the context cortex system independently

## Running Specific Tests

```bash
# Run only unit tests
pytest agent/tests/test_orchestrator_cortex_unit.py

# Run only integration tests
pytest agent/tests/test_orchestrator_cortex_integration.py -m integration

# Run with verbose output
pytest agent/tests/ -v
```

