"""
backend.app.core.logging
~~~~~~~~~~~~~~~~~~~~~~~~
Structured logging setup using structlog.

Call configure_logging() once at application startup (in main.py).
After that, all modules should use:

    import logging
    logger = logging.getLogger(__name__)

Or for structlog-native usage:
    import structlog
    logger = structlog.get_logger(__name__)
"""
from __future__ import annotations

import logging
import sys
from typing import Literal

import structlog


def configure_logging(
    level: str = "INFO",
    format: Literal["json", "console"] = "console",
    log_file: str | None = None,
) -> None:
    """Configure structlog + stdlib logging.

    Parameters
    ----------
    level    : Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    format   : 'json' for machine-readable logs, 'console' for human-readable.
    log_file : Optional file path to write logs to (in addition to stdout).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Shared processors for both structlog and stdlib
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            *shared_processors,
            renderer,
        ],
    )

    # Root handler (stdout)
    handlers: list[logging.Handler] = [
        _make_stream_handler(formatter),
    ]

    # Optional file handler
    if log_file:
        handlers.append(_make_file_handler(log_file, formatter))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(log_level)

    # Silence overly verbose third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _make_stream_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    return handler


def _make_file_handler(path: str, formatter: logging.Formatter) -> logging.FileHandler:
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(formatter)
    return handler


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Convenience wrapper — returns a structlog bound logger."""
    return structlog.get_logger(name)
