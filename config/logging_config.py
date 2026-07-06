"""Logging setup (Section 0).

Structured enough to support the REG-07 audit-trail requirement ("all
match communications are logged and subject to audit") without adding
a dependency: a small JSON formatter built on stdlib `logging` + `json`.
"""
import json
import logging
import sys
from typing import Any, Dict

_SECRET_KEY_SUFFIXES = ("_API_KEY", "_TOKEN", "_SECRET")
_STANDARD_LOGRECORD_ATTRS = frozenset(logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys())


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOGRECORD_ATTRS and key not in payload:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO", fmt: str = "plain") -> None:
    """Idempotent: safe to call from every Agent() construction without
    accumulating duplicate handlers (e.g. across tests)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    if fmt.lower() == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
    root.addHandler(handler)


def redact(value: str, visible_chars: int = 4) -> str:
    """Redact a secret for safe logging: keep a few trailing characters
    for debugging ('which key is this') without exposing the secret."""
    if not value:
        return ""
    if len(value) <= visible_chars:
        return "*" * len(value)
    return "*" * (len(value) - visible_chars) + value[-visible_chars:]


def redact_env_snapshot(env: Dict[str, str]) -> Dict[str, str]:
    """Redact values for any key that looks like a secret before logging
    an environment snapshot (e.g. a startup debug log)."""
    return {
        key: (redact(value) if key.upper().endswith(_SECRET_KEY_SUFFIXES) else value)
        for key, value in env.items()
    }
