"""Configuration layer.

Two distinct things live here:

* ``Settings`` - runtime/environment configuration (paths, log level, API
  behaviour). Sourced from environment variables with a ``.env`` fallback.
* ``load_model_config`` / ``load_risk_policy`` / ``load_stress_scenarios`` -
  the YAML files under ``config/`` that define the feature contract, the credit
  policy assumptions and the stress scenarios.

Keeping the risk policy in YAML rather than in code is deliberate: every
assumption behind LGD, EAD and the approve/decline cut-off has to be auditable
and changeable without a code deploy.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class Settings(BaseSettings):
    """Runtime settings, overridable by environment variable."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CREDIT_RISK_",
        extra="ignore",
    )

    # Paths
    data_path: Path = PROJECT_ROOT / "data" / "Loan_Default.csv"
    # Committed 5,000-row stratified sample. The full 28MB dataset is not in
    # version control, so this is what portfolio and stress endpoints fall back
    # to on a deployed instance built straight from the repository.
    portfolio_sample_path: Path = PROJECT_ROOT / "data" / "portfolio_sample.csv"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"
    reports_dir: Path = PROJECT_ROOT / "reports"
    config_dir: Path = CONFIG_DIR

    # Which model version the API serves. "latest" resolves via the registry.
    model_version: str = "latest"

    # API
    api_title: str = "Credit Risk Decisioning API"
    api_version: str = "1.0.0"
    log_level: str = "INFO"
    json_logs: bool = True
    # Comma-separated. Defaults to localhost only - never "*" with credentials.
    cors_origins: str = "http://localhost:3000,http://localhost:8000"
    max_batch_size: int = 1000

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@functools.lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings accessor. Used as a FastAPI dependency."""
    return Settings()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@functools.lru_cache(maxsize=1)
def load_model_config(path: Path | None = None) -> dict[str, Any]:
    """Feature contract, candidate models and validation settings."""
    return _load_yaml(path or CONFIG_DIR / "model.yaml")


@functools.lru_cache(maxsize=1)
def load_risk_policy(path: Path | None = None) -> dict[str, Any]:
    """Grade scale, LGD/EAD assumptions and decision economics."""
    return _load_yaml(path or CONFIG_DIR / "risk_policy.yaml")


@functools.lru_cache(maxsize=1)
def load_stress_scenarios(path: Path | None = None) -> dict[str, Any]:
    """Stress scenario definitions."""
    return _load_yaml(path or CONFIG_DIR / "stress_scenarios.yaml")


def excluded_columns(model_cfg: dict[str, Any] | None = None) -> list[str]:
    """Every column excluded from the model, across all exclusion reasons.

    Used by the training pipeline and asserted by the leakage regression test.
    """
    cfg = model_cfg or load_model_config()
    excl = cfg["exclusions"]
    return [c for group in excl.values() for c in group]
