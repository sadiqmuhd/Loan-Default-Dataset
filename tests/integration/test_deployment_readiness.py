"""Checks that the repository can actually serve a deploy.

These exist because of a real failure: the model artifact was excluded by a
stale .gitignore rule, so the deployed service started with no model and its
health check failed forever. Everything passed locally, because locally the file
was on disk. What mattered was whether it was *tracked by git*.
"""

from __future__ import annotations

import subprocess

import pytest

from loan_default.config import PROJECT_ROOT, get_settings
from loan_default.models.registry import ModelRegistry


def tracked_files(pattern: str) -> list[str]:
    """Files git actually tracks. Not the same question as os.path.exists."""
    result = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.skip("not a git repository")
    return [line for line in result.stdout.splitlines() if line.strip()]


def test_a_model_artifact_is_tracked_by_git():
    """The deploy builds from a clone, so an untracked model does not exist to it."""
    artifacts = tracked_files("artifacts/*/model.joblib")
    assert artifacts, (
        "No model.joblib is tracked by git. The service will start without a "
        "model and fail its health check. Check .gitignore is not excluding "
        "artifacts/."
    )


def test_the_version_named_by_latest_is_the_one_committed():
    """LATEST must point at an artifact that exists in the repository."""
    latest = (get_settings().artifacts_dir / "LATEST").read_text(encoding="utf-8").strip()
    tracked = tracked_files(f"artifacts/{latest}/model.joblib")
    assert tracked, (
        f"artifacts/LATEST names {latest!r}, but that version's model.joblib is "
        "not tracked by git. This is what happens when a .gitignore whitelist "
        "goes stale after a retrain."
    )


def test_model_metadata_and_metrics_are_tracked():
    """Provenance travels with the model or it is not governance."""
    latest = (get_settings().artifacts_dir / "LATEST").read_text(encoding="utf-8").strip()
    for filename in ("metadata.json", "metrics.json"):
        assert tracked_files(f"artifacts/{latest}/{filename}"), f"{filename} is not committed"


def test_the_committed_artifact_loads_and_scores():
    """Tracked is necessary but not sufficient - it also has to work."""
    from loan_default.api.schemas.requests import EXAMPLE_VALUES
    from loan_default.api.service import ScoringService

    model, metadata, metrics = ModelRegistry(get_settings().artifacts_dir).load("latest")
    service = ScoringService(model, metadata, metrics)
    record = {c: EXAMPLE_VALUES[c] for c in metadata.feature_columns if c in EXAMPLE_VALUES}
    result = service.assess_one(record, explain=False)
    assert 0.0 <= result["probability_of_default"] <= 1.0


def test_deployment_config_binds_the_injected_port():
    """Hardcoding a port makes the service unreachable behind Railway's router."""
    for filename in ("railway.toml", "nixpacks.toml", "Procfile"):
        text = (PROJECT_ROOT / filename).read_text(encoding="utf-8")
        assert "$PORT" in text, f"{filename} does not bind $PORT"
        assert "0.0.0.0" in text, f"{filename} does not bind 0.0.0.0"


def test_start_command_uses_a_single_worker():
    """Each worker holds its own model and SHAP explainer at roughly 250MB, so
    two do not fit in a 512MB instance."""
    text = (PROJECT_ROOT / "railway.toml").read_text(encoding="utf-8")
    assert "--workers 1" in text


def test_the_training_dataset_is_not_committed():
    """28MB of CSV has no place in a deploy, and the sample covers the endpoints."""
    assert not tracked_files("data/Loan_Default.csv")
    assert tracked_files("data/portfolio_sample.csv")
