FROM ollama/ollama:latest

ARG OLLAMA_MODEL=qwen3.5:4b

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OPENAI_API_KEY=ollama \
    OPENAI_BASE_URL=http://localhost:11434/v1 \
    OPENAI_MODEL=${OLLAMA_MODEL} \
    LANGCHAIN_TRACING_V2=false

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create isolated Python venv and install dependencies
COPY pyproject.toml ./
COPY src ./src
COPY pc_parts.db ./pc_parts.db

RUN python3 -m venv /app/.venv \
    && /app/.venv/bin/pip install --no-cache-dir --upgrade pip \
    && /app/.venv/bin/pip install --no-cache-dir .

# Bake the Ollama LLM model directly into the image
RUN ollama serve & \
    pid=$!; \
    until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do sleep 1; done; \
    ollama pull "${OLLAMA_MODEL}"; \
    kill "$pid" && wait "$pid" || true

COPY docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD []
