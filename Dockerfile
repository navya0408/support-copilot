# Build from the repo root:  docker build -t support-copilot .
FROM python:3.13-slim
WORKDIR /app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend backend
COPY knowledge_base knowledge_base

WORKDIR /app/backend
# Pre-build the vector index so the first request is fast
RUN python build_index.py

ENV PORT=8000
EXPOSE 8000
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}
