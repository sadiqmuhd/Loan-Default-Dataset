"""Model-validation tests: grade rank-ordering, stress behaviour and drift detection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from credit_risk.config import load_stress_scenarios
from credit_risk.data.loader import load_dataset
from credit_risk.monitoring.drift import (
    calibration_drift,
    feature_drift,
    population_stability_index,
    prediction_drift,
)
from credit_risk.risk import stress
from credit_risk.risk.grades import grade_summary, is_monotonic
from tests.conftest import requires_data, requires_model

pytestmark = [requires_model, requires_data]


@pytest.fixture(scope="module")
def scored_test_set(scoring_service):
    """Hold-out set scored once, reusing the training split."""
    dataset = load_dataset()
    _, X_test, _, y_test = train_test_split(
        dataset.X, dataset.y, test_size=0.2, stratify=dataset.y, random_state=42
    )
    frame = X_test[scoring_service.metadata.feature_columns]
    return frame, y_test, scoring_service.predict_pd(frame)


# ------------------------------------------------------------------ grades


def test_grades_rank_order_correctly(scored_test_set):
    """THE core validation test for a grading system.

    Observed default rate must increase monotonically from A through G. If it
    does not, the grades are not usable for lending decisions.
    """
    _, y_test, predictions = scored_test_set
    summary = grade_summary(predictions, y_test)
    assert is_monotonic(summary), (
        "Observed default rate is not monotonic across grades:\n"
        f"{summary[['grade', 'n', 'observed_default_rate']].to_string(index=False)}"
    )


def test_predicted_and_observed_default_rates_agree_by_grade(scored_test_set):
    """Calibration check at grade level, where it actually matters."""
    _, y_test, predictions = scored_test_set
    summary = grade_summary(predictions, y_test)
    material = summary[summary["n"] >= 200]
    gap = (material["mean_pd"] - material["observed_default_rate"]).abs()
    assert gap.max() < 0.05, (
        f"Predicted and observed default rates diverge by up to {gap.max():.4f}"
    )


def test_grades_span_a_meaningful_range(scored_test_set):
    _, y_test, predictions = scored_test_set
    summary = grade_summary(predictions, y_test)
    assert len(summary) >= 5, "Grades collapse into too few buckets to be useful."
    rates = summary["observed_default_rate"]
    assert rates.max() - rates.min() > 0.3


# ------------------------------------------------------------------ stress


def test_shocks_move_risk_in_the_right_direction(scoring_service, scored_test_set):
    """An income shock must RAISE modelled risk.

    This test is impossible to pass on the original model: `income` had exactly
    zero feature importance there, so shocking it moved PD by 0.0000.
    """
    frame, _, base_predictions = scored_test_set
    sample = frame.head(2000)
    base = scoring_service.predict_pd(sample).mean()

    shocked = stress.apply_shocks(sample, {"income": -0.25})
    stressed = scoring_service.predict_pd(shocked).mean()

    assert stressed > base, (
        f"A 25% income shock did not increase PD (base {base:.4f}, stressed "
        f"{stressed:.4f}). The model is not responding to income."
    )


def test_property_shock_propagates_to_ltv(scoring_service, scored_test_set):
    """The collateral channel must stay internally consistent: a property value
    fall raises LTV, which is a model input."""
    frame, _, _ = scored_test_set
    sample = frame.head(500)
    shocked = stress.apply_shocks(sample, {"property_value": -0.30}, recompute_ltv=True)

    assert (shocked["property_value"] < sample["property_value"]).all()
    assert (shocked["LTV"] > sample["LTV"]).all()


def test_severe_stress_costs_more_than_moderate(scoring_service, scored_test_set):
    frame, _, _ = scored_test_set
    sample = frame.head(2000)
    config = load_stress_scenarios()
    results = {
        r.name: r
        for r in stress.run_all_scenarios(
            scoring_service.model, sample, config, scoring_service.policy
        )
    }
    assert results["base"].delta_expected_loss == pytest.approx(0.0, abs=1e-6)
    assert results["moderate"].delta_expected_loss > 0
    assert results["severe"].delta_expected_loss > results["moderate"].delta_expected_loss


def test_severe_stress_pushes_borrowers_underwater(scoring_service, scored_test_set):
    frame, _, _ = scored_test_set
    sample = frame.head(2000)
    config = load_stress_scenarios()
    results = {
        r.name: r
        for r in stress.run_all_scenarios(
            scoring_service.model, sample, config, scoring_service.policy
        )
    }
    assert results["severe"].n_underwater > results["base"].n_underwater


def test_rate_variables_cannot_be_shocked():
    """Interest rate shocks must be refused: those columns are excluded as
    leakage, so the model has no rate sensitivity to stress. Pretending
    otherwise would be fabricated sophistication."""
    frame = pd.DataFrame([{"income": 5000.0, "loan_amount": 100_000.0}])
    with pytest.raises(ValueError, match="not shockable"):
        stress.apply_shocks(frame, {"rate_of_interest": 0.02})


def test_stress_results_carry_limitations():
    assert any("CCAR" in limitation for limitation in stress.LIMITATIONS)
    assert any("time dimension" in limitation for limitation in stress.LIMITATIONS)


# ------------------------------------------------------------------- drift


def test_psi_of_identical_distributions_is_zero():
    series = pd.Series(np.random.default_rng(0).normal(size=5000))
    psi, _ = population_stability_index(series, series.copy())
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(0, 1, 5000))
    shifted = pd.Series(rng.normal(1.5, 1, 5000))
    psi, _ = population_stability_index(reference, shifted)
    assert psi > 0.25, f"PSI {psi:.4f} should flag a 1.5-sigma shift as significant"


def test_psi_handles_categoricals():
    reference = pd.Series(["a"] * 700 + ["b"] * 300)
    current = pd.Series(["a"] * 300 + ["b"] * 700)
    psi, detail = population_stability_index(reference, current)
    assert psi > 0.25
    assert len(detail) == 2


def test_simulated_shock_is_detected_by_the_drift_monitor(scoring_service, scored_test_set):
    """End-to-end demonstration, clearly a simulation.

    This dataset has no time axis (year == 2019 throughout), so there is no real
    production drift to observe. The detector is instead demonstrated against a
    deliberately perturbed sample.
    """
    frame, _, _ = scored_test_set
    reference = frame.head(3000)
    perturbed = stress.apply_shocks(reference, {"income": -0.30, "property_value": -0.25})

    results = {r.feature: r for r in feature_drift(reference, perturbed)}

    # A 30% income shock clears the 0.25 "significant" band outright. A 25%
    # property shock lands in the 0.10-0.25 "moderate" band, because that column
    # is wide and right-skewed so a relative shift moves less probability mass
    # across quantile boundaries. Both must alarm; only income must be severe.
    assert results["income"].severity == "significant"
    assert results["property_value"].psi > 0.10
    assert results["property_value"].severity in {"moderate", "significant"}
    assert results["loan_purpose"].severity == "stable", "an unshocked column must not alarm"

    shift = prediction_drift(
        scoring_service.predict_pd(reference), scoring_service.predict_pd(perturbed)
    )
    assert shift.psi > 0.10


def test_calibration_drift_reports_no_alert_on_healthy_model(scored_test_set):
    _, y_test, predictions = scored_test_set
    report = calibration_drift(y_test, predictions)
    assert report["alert"] is False, (
        f"Calibration alert fired on the accepted model: gap {report['overall_gap']:.4f}"
    )


def test_calibration_drift_alerts_when_pd_is_systematically_wrong(scored_test_set):
    _, y_test, predictions = scored_test_set
    report = calibration_drift(y_test, np.clip(np.asarray(predictions) * 0.3, 0, 1))
    assert report["alert"] is True


def test_stress_results_report_extrapolation_confidence(scoring_service, scored_test_set):
    """A dramatic stress number must come with a measure of how far outside the
    observed distribution it was produced.

    Collateral shocks dominate this model's stress response, and they do so by
    pushing the book into a high-LTV region seen in only ~1.2% of training rows.
    Reporting the loss without that context would overstate its reliability.
    """
    frame, _, _ = scored_test_set
    sample = frame.head(2000)
    results = {
        r.name: r
        for r in stress.run_all_scenarios(
            scoring_service.model, sample, load_stress_scenarios(), scoring_service.policy
        )
    }

    base = results["base"].extrapolation
    severe = results["severe"].extrapolation

    assert base["confidence"] == "high", "the unstressed book must be inside its own envelope"
    assert severe["confidence"] == "low", (
        "a 30% collateral shock pushes most of the book outside the observed "
        "range and must be flagged as an extrapolation"
    )
    assert severe["max_share_outside_envelope"] > base["max_share_outside_envelope"]
    assert "LTV" in severe


def test_extrapolation_confidence_degrades_monotonically(scoring_service, scored_test_set):
    frame, _, _ = scored_test_set
    sample = frame.head(2000)
    shares = []
    for magnitude in (0.0, -0.10, -0.20, -0.30):
        stressed = (
            stress.apply_shocks(sample, {"property_value": magnitude}) if magnitude else sample
        )
        diagnostics = stress.extrapolation_diagnostics(sample, stressed)
        shares.append(diagnostics["max_share_outside_envelope"])
    assert shares == sorted(shares), f"larger shocks must not reduce extrapolation: {shares}"
