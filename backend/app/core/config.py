"""
backend.app.core.config
~~~~~~~~~~~~~~~~~~~~~~~
Application settings using pydantic-settings.

Settings are loaded with this priority (highest → lowest):
  1. Environment variables
  2. .env file
  3. Default values defined here

Usage
-----
    from backend.app.core.config import get_settings
    settings = get_settings()
    print(settings.app_port)
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Root of the repository (two levels up from this file)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CONFIGS_DIR = _REPO_ROOT / "configs"


def _load_yaml_config(env: str) -> dict:
    """Load and merge default.yaml with the env-specific override."""
    default_path = _CONFIGS_DIR / "default.yaml"
    env_path = _CONFIGS_DIR / f"{env}.yaml"

    config: dict = {}
    if default_path.exists():
        with open(default_path, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        # Deep-merge override on top of default
        _deep_merge(config, override)

    return config


def _deep_merge(base: dict, override: dict) -> None:
    """In-place deep merge of override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value


class Settings(BaseSettings):
    """Application-wide settings.

    All fields can be overridden via environment variables (uppercase).
    Example: APP_PORT=9000 overrides app_port.
    """

    model_config = SettingsConfigDict(
        env_file=str(_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    app_debug: bool = Field(default=False, alias="APP_DEBUG")

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )
    log_format: Literal["json", "console"] = Field(default="console", alias="LOG_FORMAT")

    # Simulation defaults
    default_num_hands: int = Field(default=1_000, alias="DEFAULT_NUM_HANDS")
    default_strategy: str = Field(default="basic", alias="DEFAULT_STRATEGY")
    default_num_decks: int = Field(default=6, alias="DEFAULT_NUM_DECKS")

    # API
    api_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:3000,http://127.0.0.1:3000"
    docs_url: str | None = "/docs"
    redoc_url: str | None = "/redoc"

    @field_validator("app_env", mode="before")
    @classmethod
    def normalise_env(cls, v: str) -> str:
        return v.lower()

    @classmethod
    def from_yaml_and_env(cls) -> "Settings":
        """Create Settings by loading YAML config first, then env variables."""
        env = os.getenv("APP_ENV", "development").lower()
        yaml_cfg = _load_yaml_config(env)

        # Flatten nested YAML into env-style keys for pydantic
        flat: dict[str, object] = {}
        if "app" in yaml_cfg:
            a = yaml_cfg["app"]
            flat["APP_ENV"] = a.get("env", "development")
            flat["APP_HOST"] = a.get("host", "0.0.0.0")
            flat["APP_PORT"] = a.get("port", 8000)
            flat["APP_DEBUG"] = a.get("debug", False)
        if "logging" in yaml_cfg:
            lg = yaml_cfg["logging"]
            flat["LOG_LEVEL"] = lg.get("level", "INFO")
            flat["LOG_FORMAT"] = lg.get("format", "console")
        if "simulation" in yaml_cfg:
            sim = yaml_cfg["simulation"]
            flat["DEFAULT_NUM_HANDS"] = sim.get("default_num_hands", 1000)
            flat["DEFAULT_STRATEGY"] = sim.get("default_strategy", "basic")
            flat["DEFAULT_NUM_DECKS"] = sim.get("default_num_decks", 6)

        # Env vars override YAML
        for key, value in flat.items():
            if key not in os.environ:
                os.environ[key] = str(value)

        return cls()

    def __str__(self) -> str:
        return (
            f"Settings(env={self.app_env}, host={self.app_host}, "
            f"port={self.app_port}, log_level={self.log_level})"
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere instead of instantiating directly."""
    return Settings.from_yaml_and_env()
