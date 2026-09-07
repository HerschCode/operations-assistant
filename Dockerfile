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
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import httpx; httpx.get('http://localhost:8001/health', timeout=3).raise_for_status()" || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
