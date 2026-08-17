"""
tests/unit/core/test_config.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for the application settings / config system.
"""
from __future__ import annotations

import os

import pytest


class TestSettings:
    def test_settings_loads(self) -> None:
        """Settings object can be created without raising."""
        from backend.app.core.config import get_settings
        settings = get_settings()
        assert settings is not None

    def test_settings_has_required_fields(self) -> None:
        from backend.app.core.config import get_settings
        s = get_settings()
        assert hasattr(s, "app_env")
        assert hasattr(s, "app_host")
        assert hasattr(s, "app_port")
        assert hasattr(s, "log_level")

    def test_default_environment_is_development(self) -> None:
        """In tests we always run in development mode."""
        from backend.app.core.config import get_settings
        s = get_settings()
        assert s.app_env == "development"

    def test_app_port_is_integer(self) -> None:
        from backend.app.core.config import get_settings
        s = get_settings()
        assert isinstance(s.app_port, int)

    def test_log_level_valid(self) -> None:
        from backend.app.core.config import get_settings
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        s = get_settings()
        assert s.log_level in valid_levels

    def test_api_prefix_starts_with_slash(self) -> None:
        from backend.app.core.config import get_settings
        s = get_settings()
        assert s.api_prefix.startswith("/")

    def test_settings_str_representation(self) -> None:
        from backend.app.core.config import get_settings
        s = get_settings()
        text = str(s)
        assert "Settings" in text
