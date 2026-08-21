"""Drift endpoint tests, including simulated population shifts.

A monitoring check that only ever returns "no drift" is indistinguishable from
one that is broken, so these tests deliberately shift the population and assert
the endpoint notices.

Batches are drawn from real rows rather than copies of one application. A batch
of identical rows is a degenerate distribution and scores enormous PSI on every
feature at once, which would make the tests pass for the wrong reason and hide
whether the shift was actually detected.
"""

from __future__ import annotations

import pandas as pd
import pytest

from loan_default.config import PROJECT_ROOT
from tests.conftest import requires_model

pytestmark = requires_model

SAMPLE_PATH = PROJECT_ROOT / "data" / "portfolio_sample.csv"
SEVERITIES = {"stable", "moderate", "significant"}


@pytest.fixture(scope="module")
def real_rows(app_client) -> list[dict]:
    """Real applications, reduced to the fields the API accepts."""
    if not SAMPLE_PATH.exists():
        pytest.skip("portfolio sample not present")

    from loan_default.api.schemas.requests import validate_application

    metadata = app_client.get("/v1/model/metadata").json()
    columns = metadata["feature_columns"]
    frame = pd.read_csv(SAMPLE_PATH)
    frame = frame[[c for c in columns if c in frame.columns]].dropna()

    # Keep only rows the API actually accepts. A handful of sample rows fall
    # outside the contract's numeric bounds; including them would make the
    # row counts below off-by-a-few and obscure a genuine failure.
    accepted = [
        row for row in frame.to_dict(orient="records") if validate_application(row)[1] is None
    ]
    if len(accepted) < 200:
        pytest.skip("not enough valid rows in the sample")
    return accepted[:400]


def _drift(client, applications):
    return client.post("/v1/monitoring/drift", json={"applications": applications})


def _by_feature(body) -> dict:
    return {e["feature"]: e for e in body["feature_drift"]}


# ------------------------------------------------------------------- shape


def test_unshifted_batch_is_accepted(app_client, real_rows):
    response = _drift(app_client, real_rows)
    assert response.status_code == 200
    body = response.json()
    assert body["n_rows_compared"] == len(real_rows)
    assert body["n_baseline_rows"] > 0


def test_every_reported_feature_has_a_psi_and_severity(app_client, real_rows):
    body = _drift(app_client, real_rows).json()
    assert body["feature_drift"], "no features were compared"
    for entry in body["feature_drift"]:
        assert entry["psi"] >= 0.0
        assert entry["severity"] in SEVERITIES
        assert entry["kind"] in {"numeric", "categorical"}


def test_feature_drift_is_sorted_worst_first(app_client, real_rows):
    body = _drift(app_client, real_rows).json()
    psis = [e["psi"] for e in body["feature_drift"]]
    assert psis == sorted(psis, reverse=True)


def test_undrifted_sample_stays_mostly_stable(app_client, real_rows):
    """Rows from the same population as training should not look shifted.

    A check that fires on everything is as useless as one that fires on nothing.
    """
    body = _drift(app_client, real_rows).json()
    significant = [e["feature"] for e in body["feature_drift"] if e["severity"] == "significant"]
    assert not significant, f"unshifted data flagged as drifted: {significant}"


# ------------------------------------------------------- simulated shifts


def test_an_income_shock_is_detected(app_client, real_rows):
    """SIMULATED. Income cut 40% across the batch."""
    shifted = [{**row, "income": row["income"] * 0.6} for row in real_rows]
    body = _drift(app_client, shifted).json()

    income = _by_feature(body)["income"]
    assert income["severity"] == "significant", (
        f"A 40% income shock produced PSI {income['psi']:.3f} "
        f"({income['severity']}). The check is not working."
    )


def test_an_income_shock_does_not_flag_untouched_features(app_client, real_rows):
    """Drift should localise to what actually moved, or it cannot be acted on."""
    shifted = [{**row, "income": row["income"] * 0.6} for row in real_rows]
    by_feature = _by_feature(_drift(app_client, shifted).json())

    assert by_feature["income"]["severity"] == "significant"
    for untouched in ("term", "Credit_Score", "loan_amount"):
        if untouched in by_feature:
            assert by_feature[untouched]["severity"] == "stable", (
                f"{untouched} was not shifted but reported as {by_feature[untouched]['severity']}"
            )


def test_a_property_value_shock_is_detected(app_client, real_rows):
    """SIMULATED. Property values down 30%, matching the severe stress scenario."""
    shifted = [
        {
            **row,
            "property_value": row["property_value"] * 0.7,
            "LTV": min(row["LTV"] / 0.7, 200.0),
        }
        for row in real_rows
    ]
    by_feature = _by_feature(_drift(app_client, shifted).json())
    assert by_feature["property_value"]["severity"] in {"moderate", "significant"}
    assert by_feature["LTV"]["severity"] in {"moderate", "significant"}


def test_prediction_drift_rises_when_the_population_deteriorates(app_client, real_rows):
    """PSI on the PD distribution catches shifts no single feature reveals."""
    stressed = [
        {
            **row,
            "dtir1": min(row["dtir1"] * 1.4, 61.0),
            "LTV": min(row["LTV"] * 1.25, 200.0),
            "income": row["income"] * 0.75,
        }
        for row in real_rows
    ]
    drift = _drift(app_client, stressed).json()["prediction_drift"]
    assert drift["current_mean_pd"] > drift["reference_mean_pd"], (
        "A deteriorated population should score worse than the training baseline"
    )
    assert drift["psi"] > 0.10


def test_prediction_drift_is_low_for_an_unshifted_sample(app_client, real_rows):
    drift = _drift(app_client, real_rows).json()["prediction_drift"]
    assert drift["psi"] < 0.25


# ------------------------------------------------------------------ errors


def test_batch_of_only_invalid_rows_is_rejected(app_client, example_application):
    bad = {**example_application, "credit_type": "NOT_A_REAL_VALUE"}
    assert _drift(app_client, [bad, bad]).status_code == 422


def test_invalid_rows_are_skipped_not_fatal(app_client, real_rows, example_application):
    """One bad row should not stop a population comparison."""
    bad = {**example_application, "Region": "NOWHERE"}
    response = _drift(app_client, [*real_rows, bad])
    assert response.status_code == 200
    assert response.json()["n_rows_compared"] == len(real_rows)


def test_empty_batch_is_rejected_by_schema(app_client):
    assert _drift(app_client, []).status_code == 422


# ------------------------------------------------------------- governance


def test_response_states_what_psi_does_not_measure(app_client, real_rows):
    """Input drift is not performance decay, and the response must say so."""
    body = _drift(app_client, real_rows).json()
    assert "psi_bands" in body
    assert "outcome" in body["note"].lower()


def test_drift_is_reported_against_the_serving_model_version(app_client, real_rows):
    """A baseline from a different model would make the numbers meaningless."""
    body = _drift(app_client, real_rows).json()
    metadata = app_client.get("/v1/model/metadata").json()
    assert body["model_version"] == metadata["model_version"]
