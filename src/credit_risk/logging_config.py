"""Structured logging.

JSON by default, because a credit decisioning service needs a machine-readable
audit trail, not ``print()`` statements. Every log record emitted while handling
a request carries the request id injected by the API middleware, so a decision
can be reconstructed end to end.
"""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Set by the API middleware; read by the log formatter.
request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "taskName",
    "message",
    "asctime",
}


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if (rid := request_id_var.get()) is not None:
            payload["request_id"] = rid
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        # Anything passed via logger.info(..., extra={...}) rides along.
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value

        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Readable single-line format, for local development."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-7s %(name)-38s %(message)s",
            datefmt="%H:%M:%S",
        )


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Install a single stdout handler on the root logger. Idempotent."""
    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else HumanFormatter())
    root.addHandler(handler)
    root.setLevel(level.upper())

    # These are noisy and rarely useful at INFO.
    for noisy in ("urllib3", "matplotlib", "numexpr", "shap"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
