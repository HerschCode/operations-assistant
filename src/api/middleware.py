import time
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.observability.logging_config import get_logger

logger = get_logger("api.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Logs every request: path, status, latency. Assigns a request_id so a single
    request's log line can be correlated with whatever else it touches (a future
    addition could thread this into the analytics/DB layer too)."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        start = time.monotonic()

        response = await call_next(request)

        duration_ms = round((time.monotonic() - start) * 1000, 1)
        logger.info(
            "request handled",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": duration_ms,
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response
