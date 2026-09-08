FROM python:3.12-slim

WORKDIR /app

# Install deps first, separate from copying source -- lets Docker cache this layer
# across rebuilds when only application code changes, not dependencies.
#
# Retry loop: found necessary by actually building this image, not by inspection --
# a real host networking issue (Docker Desktop/WSL2, confirmed via
# operations-performance's identical Dockerfile fix -- see that project's PLAN.md)
# intermittently fails large pip installs with SSLError "record layer failure" at a
# random point. An explicit success flag and a final check that actually fails the
# build if every attempt fails -- a bare `CMD && break; sleep N` loop's own exit
# status is its LAST command's, not the retried command's, so it would silently
# report success and ship an image missing packages if every attempt failed. That
# exact bug was caught and fixed in operations-performance's Dockerfile first.
COPY requirements.txt .
RUN success=0; \
    for i in 1 2 3 4 5 6 7 8; do \
      if pip install --no-cache-dir --default-timeout=100 --retries=5 -r requirements.txt; then \
        success=1; break; \
      fi; \
      echo "pip install attempt $i failed, retrying..."; sleep 5; \
    done; \
    if [ "$success" -ne 1 ]; then echo "pip install failed after all attempts"; exit 1; fi

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
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8001}"]
