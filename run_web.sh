#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting HardwarePCAgent Web UI at http://localhost:8000"
exec .venv/bin/uvicorn src.web:web_app --host 0.0.0.0 --port 8000
