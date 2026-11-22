#!/usr/bin/env bash
set -euo pipefail

MODEL="${OPENAI_MODEL:-qwen3.5:4b}"

echo "[entrypoint] Starting Ollama server in background..."
ollama serve &
OLLAMA_PID=$!

cleanup() {
    echo "[entrypoint] Shutting down Ollama..."
    kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[entrypoint] Waiting for Ollama API on http://localhost:11434..."
for _ in $(seq 1 60); do
    if curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 1
done

if ! curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; then
    echo "[entrypoint] ERROR: Ollama failed to start in time." >&2
    exit 1
fi

if ! ollama list | awk '{print $1}' | grep -qx "$MODEL"; then
    echo "[entrypoint] Model '$MODEL' not found locally, pulling..."
    ollama pull "$MODEL"
else
    echo "[entrypoint] Model '$MODEL' is ready."
fi

echo "[entrypoint] Starting HardwarePCAgent Web UI on http://0.0.0.0:8000..."
exec /app/.venv/bin/uvicorn src.web:web_app --host 0.0.0.0 --port 8000
