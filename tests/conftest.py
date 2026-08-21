"""Shared fixtures.

Note there is no ``sys.path`` manipulation here. The package is installed
(``pip install -e .``) and pytest resolves it via ``pythonpath = ["src"]`` in
pyproject.toml. The original test file did ``sys.path.insert(0, ...)`` to reach
its modules.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import pytest

from loan_default.config import PROJECT_ROOT, get_settings, load_model_config
from loan_default.models.registry import ModelRegistry

warnings.filterwarnings("ignore", category=DeprecationWarning)

DATA_PATH = PROJECT_ROOT / "data" / "Loan_Default.csv"


def _artifact_available() -> bool:
    try:
        ModelRegistry(get_settings().artifacts_dir).resolve_version("latest")
        return True
    except Exception:
        return False


requires_model = pytest.mark.skipif(
    not _artifact_available(),
    reason="No trained model artifact. Run: python -m loan_default.models.train",
)
requires_data = pytest.mark.skipif(
    not DATA_PATH.exists(),
    reason="Dataset not present. See README.md for how to obtain it.",
)


@pytest.fixture(scope="session")
def model_config() -> dict:
    return load_model_config()


@pytest.fixture(scope="session")
def raw_data() -> pd.DataFrame:
    """The full raw dataset, loaded once per session."""
    if not DATA_PATH.exists():
        pytest.skip("dataset not available")
    return pd.read_csv(DATA_PATH)


@pytest.fixture(scope="session")
def loaded_model():
    """(model, metadata, metrics) from the registry, loaded once."""
    if not _artifact_available():
        pytest.skip("no model artifact")
    return ModelRegistry(get_settings().artifacts_dir).load("latest")


@pytest.fixture(scope="session")
def app_client():
    """A TestClient with lifespan run, so the model is actually loaded."""
    from fastapi.testclient import TestClient

    from loan_default.api.main import create_app

    with TestClient(create_app()) as client:
        yield client


@pytest.fixture(scope="session")
def example_application() -> dict:
    from loan_default.api.schemas.requests import EXAMPLE_VALUES

    return dict(EXAMPLE_VALUES)


@pytest.fixture(scope="session")
def scoring_service(loaded_model):
    from loan_default.api.service import ScoringService
    from loan_default.models.explain import PredictionExplainer

    model, metadata, metrics = loaded_model
    return ScoringService(model, metadata, metrics, PredictionExplainer(model))


@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts"
    d.mkdir()
    return d
