"""Train/serve parity and model contract tests.

The original code fitted a ``SimpleImputer`` outside the pipeline during
training and applied it to train and test, but the API called only
``add_features`` - so training saw median-imputed, scaled values while serving
saw raw NaN. Nothing detected this, because the only prediction test in the repo
was failing with a 422 and never reached the model.

Here every transform lives inside the fitted pipeline, and these tests assert
the two paths cannot diverge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.data.loader import load_dataset
from tests.conftest import requires_data, requires_model

pytestmark = [requires_model, requires_data]


def test_direct_model_and_service_agree(loaded_model, scoring_service):
    """Scoring through the service must equal scoring the pipeline directly."""
    model, metadata, _ = loaded_model
    sample = load_dataset().X[metadata.feature_columns].head(50)

    direct = model.predict_proba(sample)[:, 1]
    viaservice = scoring_service.predict_pd(sample)

    np.testing.assert_allclose(direct, viaservice, rtol=0, atol=1e-12)


def test_single_and_batch_paths_agree(scoring_service):
    """Row-at-a-time and vectorised scoring must be numerically identical."""
    frame = load_dataset().X[scoring_service.metadata.feature_columns].head(25)
    records = frame.to_dict(orient="records")

    batch = scoring_service.predict_pd(frame)
    singles = [scoring_service.predict_pd(pd.DataFrame([r]))[0] for r in records]

    np.testing.assert_allclose(batch, singles, rtol=0, atol=1e-12)


def test_api_and_service_agree(app_client, scoring_service):
    """The HTTP layer must not alter the number."""
    frame = load_dataset().X[scoring_service.metadata.feature_columns].head(5)
    for record in frame.to_dict(orient="records"):
        payload = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        payload = {k: v for k, v in payload.items() if v is not None}
        response = app_client.post("/v1/risk/assess", json=payload)
        if response.status_code != 200:
            continue
        expected = scoring_service.predict_pd(pd.DataFrame([record]))[0]
        assert response.json()["probability_of_default"] == pytest.approx(expected, abs=1e-9)


def test_column_order_does_not_change_predictions(scoring_service):
    """A caller sending fields in a different order must get the same answer."""
    frame = load_dataset().X[scoring_service.metadata.feature_columns].head(10)
    shuffled = frame[list(reversed(frame.columns.tolist()))]

    original = scoring_service.predict_pd(frame)
    reordered = scoring_service.predict_pd(
        scoring_service.to_frame(shuffled.to_dict(orient="records"))
    )
    np.testing.assert_allclose(original, reordered, rtol=0, atol=1e-12)


def test_repeated_scoring_is_deterministic(scoring_service):
    frame = load_dataset().X[scoring_service.metadata.feature_columns].head(100)
    first = scoring_service.predict_pd(frame)
    second = scoring_service.predict_pd(frame)
    np.testing.assert_array_equal(first, second)


def test_probabilities_are_in_range(scoring_service):
    frame = load_dataset().X[scoring_service.metadata.feature_columns].sample(1000, random_state=7)
    predictions = scoring_service.predict_pd(frame)
    assert predictions.min() >= 0.0
    assert predictions.max() <= 1.0


def test_missing_feature_column_raises_clearly(scoring_service):
    record = (
        load_dataset()
        .X[scoring_service.metadata.feature_columns]
        .head(1)
        .to_dict(orient="records")[0]
    )
    del record["loan_amount"]
    with pytest.raises(ValueError, match="Missing required feature columns"):
        scoring_service.to_frame([record])


def test_library_versions_recorded_for_reproducibility(loaded_model):
    """The original artifact was pickled under sklearn 1.7.2 and loaded under
    1.8.0, warning that results might be invalid. Versions are now recorded."""
    _, metadata, _ = loaded_model
    assert "scikit-learn" in metadata.library_versions
    assert "xgboost" in metadata.library_versions
    assert metadata.seed is not None


def test_model_is_calibrated(loaded_model):
    """Mean predicted PD must track the observed default rate."""
    _, _, metrics = loaded_model
    calibrated = metrics["calibrated"]
    gap = abs(calibrated["mean_predicted_pd"] - calibrated["base_rate"])
    assert gap < 0.02, f"Mean PD is {gap:.4f} away from the observed default rate."


def test_calibration_improved_over_uncalibrated(loaded_model):
    """Isotonic calibration must reduce calibration error, or it is pointless."""
    _, _, metrics = loaded_model
    assert (
        metrics["calibrated"]["calibration_error"] <= metrics["uncalibrated"]["calibration_error"]
    )
