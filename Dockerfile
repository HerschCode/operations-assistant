FROM python:3.12-slim

WORKDIR /app

# Install deps first, separate from copying source -- lets Docker cache this layer
# across rebuilds when only application code changes, not dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 8001

# A genuine health check, not just "is the process up" -- curl against /health, which
# itself checks both operations-performance's API and the vector store are reachable
# (src/api/dependencies.py). A container that's "running" but can't reach either
# dependency should be reported unhealthy, not silently accepting traffic it can't serve.
# Reads $PORT (falling back to 8001 for local docker-compose use) rather than a fixed
# port, since Render/Cloud Run/most PaaS hosts assign the listen port dynamically via
# that env var and this healthcheck has to hit whatever port the process actually bound.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import os, httpx; httpx.get(f'http://localhost:{os.environ.get(\"PORT\", 8001)}/health', timeout=3).raise_for_status()" || exit 1

# Shell form (not exec-form JSON array) specifically so $PORT is expanded at container
# start -- an exec-form CMD would pass the literal string "$PORT" to uvicorn instead of
# its value. Falls back to 8001 (this project's local/docker-compose default) when PORT
# isn't set, so local `docker run` without -e PORT=... still works unchanged.
CMD uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8001}
