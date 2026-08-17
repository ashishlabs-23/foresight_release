"""
tests/unit/core/test_logging.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the logging configuration module.
"""
from __future__ import annotations

import logging


class TestConfigureLogging:
    def test_configure_logging_does_not_raise(self) -> None:
        from backend.app.core.logging import configure_logging
        configure_logging(level="WARNING", format="console")

    def test_configure_logging_json_format(self) -> None:
        from backend.app.core.logging import configure_logging
        configure_logging(level="INFO", format="json")

    def test_log_level_is_applied(self) -> None:
        from backend.app.core.logging import configure_logging
        configure_logging(level="ERROR", format="console")
        root = logging.getLogger()
        assert root.level == logging.ERROR

    def test_get_logger_returns_logger(self) -> None:
        from backend.app.core.logging import get_logger
        logger = get_logger("test.module")
        assert logger is not None

    def test_stdlib_logger_works_after_configure(self) -> None:
        from backend.app.core.logging import configure_logging
        configure_logging(level="DEBUG", format="console")
        logger = logging.getLogger("test_stdlib")
        # Should not raise
        logger.debug("test debug message")
        logger.info("test info message")
