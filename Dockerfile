FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LANGCHAIN_TRACING_V2=false

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY pc_parts.db ./pc_parts.db

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "src.web:web_app", "--host", "0.0.0.0", "--port", "8000"]
