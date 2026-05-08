"""Structured JSON logging for the heart-disease API.

A self-contained JSON formatter (no external dep) that emits one log
record per line so it slots cleanly into Loki/ELK/CloudWatch pipelines.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone

# Standard LogRecord attributes we don't want to duplicate in `extra`.
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class JsonFormatter(logging.Formatter):
    """Format LogRecords as single-line JSON objects."""

    def __init__(self, service: str = "heart-disease-api",
                 version: str = "v1.0.0") -> None:
        super().__init__()
        self.service = service
        self.version = version

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc)
                          .isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service,
            "version": self.version,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Promote any user-supplied `extra=` fields to top-level keys.
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        return json.dumps(payload, default=str)


def configure_logging(service: str = "heart-disease-api",
                      version: str = "v1.0.0") -> None:
    """Install the JSON formatter on the root logger.

    Honours ``LOG_LEVEL`` env var (default INFO). Set ``LOG_FORMAT=plain``
    to fall back to the default human-readable formatter (handy for local
    debugging).
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler(sys.stdout)

    if os.getenv("LOG_FORMAT", "json").lower() == "plain":
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    else:
        handler.setFormatter(JsonFormatter(service=service, version=version))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    # Quiet down uvicorn's duplicate access logs (we log requests ourselves).
    for noisy in ("uvicorn.access",):
        logging.getLogger(noisy).handlers = []
        logging.getLogger(noisy).propagate = False
