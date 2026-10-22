#!/usr/bin/env bash
set -e

# Activate virtual environment if present
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

python -m src.main "$@"
