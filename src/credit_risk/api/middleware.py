"""Request correlation and timing middleware."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from credit_risk.logging_config import request_id_var

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
RESPONSE_TIME_HEADER = "X-Response-Time-ms"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id and wall-clock timing to every request.

    The id is taken from an inbound ``X-Request-ID`` when present, so a trace can
    span an upstream gateway, and is echoed on the response. It is also pushed
    into a context variable, so every log record emitted while handling the
    request carries it - which is what makes the decision audit trail joinable.
    """

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        token = request_id_var.set(request_id)
        started = time.perf_counter()

        try:
            response: Response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            logger.exception(
                "unhandled error",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(elapsed, 2),
                    "event": "http_error",
                },
            )
            raise
        finally:
            request_id_var.reset(token)

        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers[RESPONSE_TIME_HEADER] = f"{elapsed:.2f}"

        # Health probes would otherwise dominate the log volume.
        if not request.url.path.startswith("/health"):
            logger.info(
                "request completed",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": round(elapsed, 2),
                    "event": "http_request",
                },
            )
        return response
