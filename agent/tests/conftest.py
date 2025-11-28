"""
Pytest configuration for agent tests.

This ensures tests use the evals/.venv environment which has all ARE dependencies.

IMPORTANT: Tests must be run with the evals/.venv activated:
    cd evals && source .venv/bin/activate && cd .. && pytest agent/tests/

Or use the provided script:
    ./run_agent_tests.sh
"""

import sys
from pathlib import Path

# Add evals directory to path so 'are' module resolves correctly
# This allows ARE code (which uses 'from are.simulation...') to work
# when running from the root directory
evals_dir = Path(__file__).resolve().parent.parent.parent / "evals"
if str(evals_dir) not in sys.path:
    sys.path.insert(0, str(evals_dir))

