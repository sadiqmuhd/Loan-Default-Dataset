"""Tests that exercise the training pipeline itself.

Every other test in this suite scores the committed artifact, which means the
code that *produces* that artifact was the only substantial part of the package
with no coverage at all. These tests run real training end to end on a
subsample, which is slow but covers the path that matters most: if training is
wrong, everything downstream is confidently wrong.

Marked ``slow``; deselect with ``pytest -m "not slow"``.
"""

from __future__ import annotations

import pytest

from loan_default.config import get_settings, load_model_config
from loan_default.data.loader import load_dataset
from loan_default.models.evaluate import confusion_at_threshold, evaluate, ks_statistic
from loan_default.models.pipeline import build_estimator, build_pipeline
from loan_default.models.train import LeakageGuardError, _assert_no_leakage_features, train
from tests.conftest import requires_data

pytestmark = [pytest.mark.slow, requires_data]


@pytest.fixture(scope="module")
def isolated_training(tmp_path_factory):
    """Point training at a scratch directory so it cannot clobber artifacts/."""
    settings = get_settings()
    original_artifacts, original_reports = settings.artifacts_dir, settings.reports_dir
    scratch = tmp_path_factory.mktemp("training")
    settings.artifacts_dir = scratch / "artifacts"
    settings.reports_dir = scratch / "reports"
    yield settings
    settings.artifacts_dir, settings.reports_dir = original_artifacts, original_reports


@pytest.fixture(scope="module")
def two_training_runs(isolated_training):
    """Train twice with the same seed. The whole point is that they match."""
    first = train(quick=True, skip_cv=True)
    second = train(quick=True, skip_cv=True)
    return first, second


def test_training_is_deterministic(two_training_runs):
    """Same seed, same data, same metrics - otherwise nothing else is reproducible.

    A model whose measured performance moves between runs cannot be governed:
    the numbers signed off at approval would not be the numbers in production.
    """
    first, second = two_training_runs
    a = first["metrics"]["calibrated"]
    b = second["metrics"]["calibrated"]

    for metric in ("roc_auc", "pr_auc", "brier_score", "ks_statistic", "mean_predicted_pd"):
        assert a[metric] == pytest.approx(b[metric], abs=1e-12), (
            f"{metric} differs between identical runs: {a[metric]} vs {b[metric]}"
        )


def test_training_records_the_data_hash(two_training_runs):
    first, _ = two_training_runs
    provenance = first["metrics"]["data_provenance"]
    assert len(provenance["data_sha256"]) == 64
    assert provenance["n_rows_used"] > 0


def test_training_produces_a_loadable_artifact(two_training_runs, isolated_training):
    from loan_default.models.registry import ModelRegistry

    first, _ = two_training_runs
    model, metadata, metrics = ModelRegistry(isolated_training.artifacts_dir).load(first["version"])
    assert metadata.model_version == first["version"]
    assert metrics["calibrated"]["roc_auc"] > 0.70

    frame = load_dataset().X[metadata.feature_columns].head(5)
    probabilities = model.predict_proba(frame)[:, 1]
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


def test_training_records_excluded_columns(two_training_runs):
    first, _ = two_training_runs
    excluded = first["metadata"]["excluded_columns"]
    assert "Interest_rate_spread" in excluded["leakage"]
    assert "Gender" in excluded["protected"]
    assert "ID" in excluded["identifiers"]


def test_training_compares_calibration_methods(two_training_runs):
    """Both candidate calibrators should be measured, not just the configured one."""
    first, _ = two_training_runs
    comparison = first["metrics"].get("calibration_comparison", {})
    assert {"isotonic", "sigmoid", "uncalibrated"} <= set(comparison)


def test_metadata_documents_assumptions_and_limitations(two_training_runs):
    first, _ = two_training_runs
    metadata = first["metadata"]
    assert metadata["assumptions"], "a governed model must state its assumptions"
    assert metadata["limitations"], "a governed model must state its limitations"
    assert any("Credit_Score" in item for item in metadata["limitations"])


# ------------------------------------------------------------ guardrail


def test_leakage_guard_rejects_excluded_columns():
    cfg = load_model_config()
    with pytest.raises(LeakageGuardError, match="Excluded columns"):
        _assert_no_leakage_features(["loan_amount", "Interest_rate_spread"], cfg)


def test_leakage_guard_rejects_missingness_indicators():
    cfg = load_model_config()
    with pytest.raises(LeakageGuardError, match="Missingness indicators"):
        _assert_no_leakage_features(["loan_amount", "LTV_missing"], cfg)


def test_leakage_guard_accepts_a_clean_feature_set():
    cfg = load_model_config()
    _assert_no_leakage_features(["loan_amount", "LTV", "dtir1", "income"], cfg)


# ------------------------------------------------------------- components


def test_pipeline_builds_for_every_configured_candidate():
    cfg = load_model_config()
    for name, params in cfg["candidates"].items():
        pipeline = build_pipeline(
            name, params, ["loan_amount", "LTV"], ["Region"], 42, cfg["feature_params"]
        )
        assert set(pipeline.named_steps) == {"features", "preprocess", "estimator"}


def test_unknown_estimator_is_rejected():
    with pytest.raises(ValueError, match="Unknown estimator"):
        build_estimator("magic_forest", {}, 42)


def test_xgboost_does_not_set_scale_pos_weight():
    """Reweighting distorts the predicted distribution, which is incoherent with
    serving the output as a probability. Imbalance is handled by calibration."""
    cfg = load_model_config()
    estimator = build_estimator("xgboost", cfg["candidates"]["xgboost"], 42)
    assert estimator.scale_pos_weight in (None, 1, 1.0)


# -------------------------------------------------------------- evaluation


def test_evaluate_reports_the_full_metric_suite():
    y_true = [0] * 800 + [1] * 200
    y_prob = [0.1] * 800 + [0.7] * 200
    report = evaluate(y_true, y_prob)

    assert report.roc_auc == pytest.approx(1.0)
    assert report.base_rate == pytest.approx(0.2)
    assert report.gini == pytest.approx(2 * report.roc_auc - 1)
    assert report.brier_score > 0
    assert report.calibration_bins


def test_evaluate_requires_both_classes():
    with pytest.raises(ValueError, match="both classes"):
        evaluate([0] * 100, [0.1] * 100)


def test_bootstrap_interval_brackets_the_point_estimate():
    rng_true = [0, 1] * 500
    rng_prob = [0.2, 0.8] * 500
    report = evaluate(rng_true, rng_prob, bootstrap_iterations=50, seed=42)
    low, high = report.roc_auc_ci
    assert low <= report.roc_auc <= high


def test_ks_statistic_is_bounded():
    y_true = [0] * 500 + [1] * 500
    assert 0.0 <= ks_statistic(y_true, [0.2] * 500 + [0.8] * 500) <= 1.0


def test_confusion_at_threshold_counts_add_up():
    y_true = [0] * 80 + [1] * 20
    y_prob = [0.1] * 80 + [0.9] * 20
    result = confusion_at_threshold(y_true, y_prob, 0.5)
    total = (
        result["true_negatives"]
        + result["false_positives"]
        + result["false_negatives"]
        + result["true_positives"]
    )
    assert total == 100
    assert result["precision"] == pytest.approx(1.0)
    assert result["recall"] == pytest.approx(1.0)
