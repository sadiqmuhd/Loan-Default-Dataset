"""Leakage regression tests. The most important tests in this repository.

The original model scored ROC-AUC 1.0000 because ``add_features`` created a
feature ``Interest_rate_spread_missing``, and ``Interest_rate_spread.isna()``
equalled the target for 148,670 of 148,670 rows - exactly 100.000%. The notebook
committed to the repository showed logistic regression, random forest and
XGBoost all at 1.0000 on every metric.

Nothing in the codebase would have caught that. These tests would have failed
immediately, and they will fail again if any leaking feature is reintroduced.
"""

from __future__ import annotations

import numpy as np
import pytest

from credit_risk.config import excluded_columns
from credit_risk.data.loader import load_dataset
from credit_risk.features.engineering import add_features
from tests.conftest import requires_data, requires_model

# Columns whose missingness encodes the outcome.
KNOWN_LEAKY = ["rate_of_interest", "Interest_rate_spread", "Upfront_charges"]


@requires_data
def test_leakage_still_present_in_raw_data(raw_data):
    """Documents the defect. If this ever fails, the source data changed.

    Interest_rate_spread.isna() is an exact copy of the target.
    """
    agreement = (raw_data["Interest_rate_spread"].isna().astype(int) == raw_data["Status"]).mean()
    assert agreement == pytest.approx(1.0, abs=1e-9), (
        f"Expected Interest_rate_spread missingness to equal Status exactly; "
        f"got {agreement:.6f} agreement."
    )


@requires_data
def test_excluded_columns_absent_from_loaded_features(model_config):
    """No excluded column survives data loading."""
    dataset = load_dataset()
    present = sorted(set(dataset.X.columns) & set(excluded_columns(model_config)))
    assert present == [], f"Excluded columns reached the feature set: {present}"


@requires_data
@pytest.mark.parametrize("column", KNOWN_LEAKY)
def test_leaky_columns_dropped(column):
    assert column not in load_dataset().X.columns


@requires_data
def test_protected_attributes_dropped():
    """ECOA / Regulation B: gender and age must never reach the model."""
    columns = load_dataset().X.columns
    for attribute in ("Gender", "age"):
        assert attribute not in columns, f"{attribute} is a protected characteristic"


def test_feature_engineering_creates_no_missingness_indicators(raw_data):
    """The specific mechanism of the original leak is now structurally impossible."""
    engineered = add_features(raw_data.head(500))
    indicators = [c for c in engineered.columns if c.endswith(("_missing", "_isna", "_is_null"))]
    assert indicators == [], (
        f"Missingness indicators are forbidden - they encoded the target in the "
        f"original model. Found: {indicators}"
    )


@requires_model
def test_model_features_contain_no_excluded_columns(loaded_model, model_config):
    _, metadata, _ = loaded_model
    overlap = sorted(set(metadata.feature_columns) & set(excluded_columns(model_config)))
    assert overlap == [], f"Model was trained on excluded columns: {overlap}"


@requires_model
def test_test_auc_below_plausible_ceiling(loaded_model, model_config):
    """THE GUARDRAIL.

    A leakage-free model on this data scores ~0.82. Anything approaching 1.0
    means a leaking feature has been reintroduced. This test failing is a
    feature, not a nuisance.
    """
    _, _, metrics = loaded_model
    auc = metrics["calibrated"]["roc_auc"]
    ceiling = float(model_config["max_plausible_auc"])
    assert auc < ceiling, (
        f"Test ROC-AUC {auc:.4f} exceeds the plausible ceiling {ceiling}. "
        "A leaking feature has almost certainly been reintroduced."
    )


@requires_model
def test_auc_is_meaningfully_better_than_random(loaded_model):
    """Guard the other direction: the model must actually discriminate."""
    _, _, metrics = loaded_model
    assert metrics["calibrated"]["roc_auc"] > 0.70


@requires_model
def test_no_single_feature_dominates(loaded_model):
    """In the original model two leakage indicators held 97.5% of importance and
    82 of 87 features had exactly zero importance. Genuine signal is distributed.
    """
    from credit_risk.models.explain import PredictionExplainer

    model, metadata, _ = loaded_model
    explainer = PredictionExplainer(model)
    if not explainer.available:
        pytest.skip("explainer unavailable for this estimator")

    dataset = load_dataset()
    importance = explainer.global_importance(dataset.X[metadata.feature_columns], sample=1000)
    total = importance["mean_abs_shap"].sum()
    assert total > 0
    top_share = importance["mean_abs_shap"].iloc[0] / total
    assert top_share < 0.50, (
        f"Top feature '{importance['feature'].iloc[0]}' holds {top_share:.1%} of total "
        "importance, which suggests leakage."
    )


@requires_data
def test_complete_case_filter_removes_second_order_leakage(model_config):
    """Missingness of these columns predicts default (AUC 0.7155 on its own),
    so no row retained may be missing them."""
    dataset = load_dataset()
    for column in model_config["complete_case_columns"]:
        if column in dataset.X.columns:
            assert not dataset.X[column].isna().any(), (
                f"{column} has nulls after the complete-case filter"
            )


@requires_data
def test_year_is_constant_so_no_temporal_validation_is_claimed(raw_data):
    """Documents why no out-of-time validation exists: there is no time axis."""
    assert raw_data["year"].nunique() == 1, (
        "year is no longer constant - out-of-time validation may now be possible "
        "and the model card's limitations should be revisited."
    )


@requires_data
def test_credit_score_is_non_predictive_noise(raw_data):
    """Documents a dataset limitation stated in the model card.

    Credit_Score has univariate ROC-AUC ~0.503 and a flat default rate across
    deciles. In a real bureau dataset this would be the strongest single
    predictor; here it appears to be randomly generated.
    """
    from sklearn.metrics import roc_auc_score

    subset = raw_data.dropna(subset=["Credit_Score", "Status"])
    auc = roc_auc_score(subset["Status"], subset["Credit_Score"])
    assert abs(auc - 0.5) < 0.02, (
        f"Credit_Score univariate AUC is {auc:.4f}. If this is no longer ~0.5 the "
        "dataset changed and the model card limitation should be revisited."
    )


@requires_model
def test_predictions_are_not_degenerate(scoring_service):
    """A leaking model produces PDs piled at 0 and 1. A real one is spread out."""
    dataset = load_dataset()
    sample = dataset.X[scoring_service.metadata.feature_columns].sample(2000, random_state=42)
    predictions = scoring_service.predict_pd(sample)

    extreme = np.mean((predictions < 0.01) | (predictions > 0.99))
    assert extreme < 0.25, (
        f"{extreme:.1%} of predictions are at the extremes, which indicates the "
        "model is separating the classes almost perfectly - a leakage signature."
    )
    assert predictions.std() > 0.05
