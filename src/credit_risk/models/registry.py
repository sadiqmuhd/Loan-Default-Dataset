"""Versioned model artifact storage.

Each trained model is written to ``artifacts/<version>/`` containing:

    model.joblib     the fitted, calibrated sklearn pipeline
    metadata.json    version, timestamps, data hash, feature contract, assumptions
    metrics.json     the full evaluation report

This is the project's governance surface: given a prediction, the model version
in the response resolves to the exact artifact, the exact training data hash and
the exact metrics that were accepted at approval time.

Artifacts are stored with ``joblib`` rather than raw ``pickle``, and the metadata
records the library versions used, because the original project shipped a pickle
written under scikit-learn 1.7.2 that emitted ``InconsistentVersionWarning`` when
loaded under 1.8.0. Loading now verifies this explicitly.
"""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost

logger = logging.getLogger(__name__)

MODEL_FILENAME = "model.joblib"
METADATA_FILENAME = "metadata.json"
METRICS_FILENAME = "metrics.json"


@dataclass
class ModelMetadata:
    """Everything needed to reproduce, audit and trace a model."""

    model_version: str
    model_type: str
    trained_at: str
    training_duration_seconds: float

    # Data provenance
    data_sha256: str
    data_source: str
    n_training_rows: int
    n_test_rows: int
    default_rate: float

    # Feature contract
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    engineered_features: list[str]
    excluded_columns: dict[str, list[str]]

    # Reproducibility
    seed: int
    python_version: str
    library_versions: dict[str, str]

    # Model detail
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    calibration_method: str = "none"
    selection_metric: str = "pr_auc"
    candidate_scores: dict[str, float] = field(default_factory=dict)

    # Governance
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def current_library_versions() -> dict[str, str]:
    return {
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


class ModelRegistry:
    """Filesystem-backed registry of versioned model artifacts."""

    def __init__(self, artifacts_dir: Path):
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ write

    def save(
        self,
        model: Any,
        metadata: ModelMetadata,
        metrics: dict[str, Any],
    ) -> Path:
        target = self.artifacts_dir / metadata.model_version
        target.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, target / MODEL_FILENAME)
        (target / METADATA_FILENAME).write_text(
            json.dumps(metadata.to_dict(), indent=2, default=str), encoding="utf-8"
        )
        (target / METRICS_FILENAME).write_text(
            json.dumps(metrics, indent=2, default=str), encoding="utf-8"
        )
        (self.artifacts_dir / "LATEST").write_text(metadata.model_version, encoding="utf-8")

        logger.info("saved model %s to %s", metadata.model_version, target)
        return target

    # ------------------------------------------------------------------- read

    def list_versions(self) -> list[str]:
        return sorted(
            p.name
            for p in self.artifacts_dir.iterdir()
            if p.is_dir() and (p / MODEL_FILENAME).exists()
        )

    def resolve_version(self, version: str = "latest") -> str:
        if version != "latest":
            return version
        pointer = self.artifacts_dir / "LATEST"
        if pointer.exists():
            resolved = pointer.read_text(encoding="utf-8").strip()
            if (self.artifacts_dir / resolved / MODEL_FILENAME).exists():
                return resolved
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError(
                f"No model artifacts in {self.artifacts_dir}. Run: python -m credit_risk.models.train"
            )
        return versions[-1]

    def load(self, version: str = "latest") -> tuple[Any, ModelMetadata, dict[str, Any]]:
        resolved = self.resolve_version(version)
        target = self.artifacts_dir / resolved
        model_path = target / MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {model_path}")

        model = joblib.load(model_path)
        metadata = ModelMetadata(
            **json.loads((target / METADATA_FILENAME).read_text(encoding="utf-8"))
        )
        metrics = json.loads((target / METRICS_FILENAME).read_text(encoding="utf-8"))

        self._warn_on_version_drift(metadata)
        return model, metadata, metrics

    @staticmethod
    def _warn_on_version_drift(metadata: ModelMetadata) -> None:
        """Log loudly if the runtime differs from the environment that trained this."""
        current = current_library_versions()
        for lib, trained_version in metadata.library_versions.items():
            running = current.get(lib)
            if running and running != trained_version:
                logger.warning(
                    "library version drift for %s: model trained with %s, running %s. "
                    "Predictions may differ from the accepted metrics.",
                    lib,
                    trained_version,
                    running,
                )


def new_version_string(prefix: str = "v") -> str:
    """UTC timestamp version, e.g. v20260821T142530Z. Sorts chronologically."""
    return prefix + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def build_metadata(**kwargs: Any) -> ModelMetadata:
    kwargs.setdefault("python_version", platform.python_version())
    kwargs.setdefault("library_versions", current_library_versions())
    kwargs.setdefault("trained_at", datetime.now(UTC).isoformat())
    return ModelMetadata(**kwargs)
