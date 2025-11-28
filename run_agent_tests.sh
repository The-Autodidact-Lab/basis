#!/bin/bash
# Run agent tests using the evals/.venv environment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVALS_DIR="$SCRIPT_DIR/evals"
VENV_PYTHON="$EVALS_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "Error: evals/.venv not found. Please set up the evals environment first."
    echo "Run: cd evals && uv sync"
    exit 1
fi

# Check if pytest is installed in venv
if ! "$VENV_PYTHON" -m pytest --version >/dev/null 2>&1; then
    echo "pytest not found in evals/.venv. Installing..."
    cd "$EVALS_DIR"
    if command -v uv &> /dev/null; then
        uv pip install pytest==8.3.4
    else
        echo "Error: uv not found. Please install pytest manually:"
        echo "  cd evals && uv pip install pytest==8.3.4"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# Run pytest using the venv's Python directly
cd "$SCRIPT_DIR"
"$VENV_PYTHON" -m pytest agent/tests/ "$@"

