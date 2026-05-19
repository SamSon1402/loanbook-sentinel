"""Application configuration via pydantic-settings.

All settings are loaded from environment variables (12-factor) or an optional
.env file at process startup. Configuration is **immutable** once loaded.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="SENTINEL_",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── runtime ───────────────────────────────────────────────────────────
    env: str = Field(default="local", description="local | staging | production")
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    # ── persistence ───────────────────────────────────────────────────────
    database_url: str = Field(
        default="sqlite+aiosqlite:///./sentinel.db",
        description="SQLAlchemy async URL.",
    )

    # ── ML ────────────────────────────────────────────────────────────────
    model_path: Path = Field(default=Path("models/anomaly.onnx"))
    anomaly_threshold: float = Field(
        default=-0.15,
        description="ONNX decision_function score below which a loan is flagged.",
    )

    # ── scanner ───────────────────────────────────────────────────────────
    scan_interval_seconds: int = Field(default=300, ge=10)
    dpd_warning_days: int = Field(default=30, ge=1)
    dpd_default_days: int = Field(default=90, ge=1)

    # ── observability ─────────────────────────────────────────────────────
    metrics_enabled: bool = Field(default=True)
    tracing_otlp_endpoint: str = Field(default="")

    # ── http ──────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors(cls, v: str | list[str]) -> list[str]:
        if isinstance(v, str):
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    @field_validator("log_level")
    @classmethod
    def _upper_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(f"invalid log level: {v}")
        return v

    @property
    def is_production(self) -> bool:
        return self.env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor — call from anywhere."""
    return Settings()
