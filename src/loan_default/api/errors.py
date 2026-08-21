"""Exception handlers.

    422  request failed schema validation - field-level detail is safe to return,
         since the allowed values come from the published data contract
    503  model not loaded or not ready
    500  anything unexpected: logged in full, but the client gets a generic
         message and a request id to quote

Raw exception text never reaches the response body.
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


def _payload(error: str, message: str, request: Request, details: list | None = None) -> dict:
    return {
        "error": error,
        "message": message,
        "request_id": _request_id(request),
        "details": details or [],
    }


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 with field-level detail.

    Returning which field failed and why is safe and genuinely useful - the
    allowed values come from the public data contract, not from internals.
    """
    details = []
    for error in exc.errors():
        location = [str(p) for p in error.get("loc", []) if p not in ("body",)]
        details.append(
            {"field": ".".join(location) or None, "message": error.get("msg", "invalid value")}
        )

    logger.info(
        "request validation failed",
        extra={"path": request.url.path, "n_errors": len(details), "event": "validation_error"},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_payload(
            "validation_error", "The request failed schema validation.", request, details
        ),
    )


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    code = {
        status.HTTP_404_NOT_FOUND: "not_found",
        status.HTTP_503_SERVICE_UNAVAILABLE: "service_unavailable",
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE: "payload_too_large",
    }.get(exc.status_code, "http_error")

    if exc.status_code >= 500:
        logger.error("http error", extra={"status_code": exc.status_code, "path": request.url.path})
    return JSONResponse(
        status_code=exc.status_code,
        content=_payload(code, str(exc.detail), request),
        headers=getattr(exc, "headers", None),
    )


async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """Domain errors raised by the scoring service - the caller's input is at fault."""
    logger.warning("domain error: %s", exc, extra={"path": request.url.path})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=_payload("invalid_input", str(exc), request),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """500 without leaking internals. Full detail goes to the log, not the client."""
    logger.exception(
        "unhandled exception",
        extra={"path": request.url.path, "exception_type": type(exc).__name__},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_payload(
            "internal_error",
            "An internal error occurred. Quote the request_id when reporting this.",
            request,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    # Starlette types handlers against bare Exception, so narrower signatures are
    # rejected by the checker even though FastAPI dispatches on the exact class.
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(HTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(ValueError, value_error_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
