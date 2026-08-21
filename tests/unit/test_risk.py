"""Credit risk methodology tests: grades, LGD proxy, EAD, EL and decision policy."""

from __future__ import annotations

import numpy as np
import pytest

from loan_default.config import load_risk_policy
from loan_default.risk.expected_loss import (
    compute_loss_components,
    expected_loss,
    exposure_at_default,
    loss_given_default,
)
from loan_default.risk.grades import assign_grade, assign_grades, grade_scale
from loan_default.risk.policy import Decision, break_even_pd, decide, revenue_rate
from loan_default.risk.portfolio import herfindahl_index

POLICY = load_risk_policy()


# ------------------------------------------------------------------ grades


def test_grade_scale_is_ordered_and_covers_unit_interval():
    scale = grade_scale(POLICY)
    bounds = [g.max_pd for g in scale]
    assert bounds == sorted(bounds)
    assert bounds[-1] > 1.0, "the worst grade must absorb PD = 1.0"


@pytest.mark.parametrize(
    ("pd_value", "expected"),
    [(0.001, "A"), (0.03, "B"), (0.08, "C"), (0.15, "D"), (0.30, "E"), (0.50, "F"), (0.90, "G")],
)
def test_assign_grade_boundaries(pd_value, expected):
    assert assign_grade(pd_value, POLICY) == expected


def test_assign_grade_handles_extremes():
    assert assign_grade(0.0, POLICY) == "A"
    assert assign_grade(1.0, POLICY) == "G"


def test_vectorised_grades_match_scalar():
    values = np.array([0.001, 0.03, 0.08, 0.15, 0.30, 0.50, 0.90])
    vector = assign_grades(values, POLICY)
    scalar = [assign_grade(v, POLICY) for v in values]
    assert list(vector) == scalar


# --------------------------------------------------------------------- EAD


def test_ead_equals_origination_amount():
    """Fully-drawn term loans: EAD is the origination amount, no CCF."""
    assert exposure_at_default([100_000.0], POLICY)[0] == pytest.approx(100_000.0)


# --------------------------------------------------------------------- LGD


def test_lgd_falls_when_collateral_is_strong():
    """A well-secured loan recovers more, so LGD is lower."""
    strong = loss_given_default([100_000.0], [500_000.0], POLICY)[0]
    weak = loss_given_default([100_000.0], [110_000.0], POLICY)[0]
    assert strong < weak


def test_lgd_respects_floor_and_ceiling():
    cfg = POLICY["lgd"]
    over_secured = loss_given_default([100_000.0], [10_000_000.0], POLICY)[0]
    assert over_secured == pytest.approx(cfg["floor"])

    unsecured = loss_given_default([1_000_000.0], [1_000.0], POLICY)[0]
    assert unsecured <= cfg["ceiling"]
    assert unsecured > 0.9


def test_lgd_formula_matches_documented_arithmetic():
    cfg = POLICY["lgd"]
    ead, collateral = 200_000.0, 250_000.0
    recoverable = collateral * (1 - cfg["distressed_sale_haircut"]) * (1 - cfg["workout_cost_rate"])
    expected = np.clip(1 - recoverable / ead, cfg["floor"], cfg["ceiling"])
    assert loss_given_default([ead], [collateral], POLICY)[0] == pytest.approx(expected)


def test_lgd_uses_fallback_without_collateral():
    result = loss_given_default([100_000.0], None, POLICY)[0]
    assert result == pytest.approx(POLICY["lgd"]["fallback_lgd"])


# ---------------------------------------------------------------------- EL


def test_expected_loss_is_product_of_components():
    result = expected_loss([0.10], [200_000.0], [250_000.0], POLICY)
    assert result["expected_loss"][0] == pytest.approx(
        result["pd"][0] * result["lgd"][0] * result["ead"][0]
    )


def test_expected_loss_scales_linearly_with_pd():
    low = expected_loss([0.05], [200_000.0], [250_000.0], POLICY)["expected_loss"][0]
    high = expected_loss([0.10], [200_000.0], [250_000.0], POLICY)["expected_loss"][0]
    assert high == pytest.approx(2 * low)


def test_zero_pd_gives_zero_expected_loss():
    assert expected_loss([0.0], [200_000.0], [250_000.0], POLICY)["expected_loss"][0] == 0.0


def test_loss_components_flags_the_lgd_method():
    with_collateral = compute_loss_components(0.1, 200_000.0, 250_000.0, POLICY)
    without = compute_loss_components(0.1, 200_000.0, None, POLICY)
    assert with_collateral.lgd_method == "collateral_proxy"
    assert without.lgd_method == "fallback_flat_rate"


# ------------------------------------------------------------------ policy


def test_break_even_pd_matches_closed_form():
    """PD* = revenue_rate / (LGD * aversion + revenue_rate)."""
    lgd = 0.25
    rate = revenue_rate(POLICY)
    aversion = POLICY["economics"]["loss_aversion_multiplier"]
    assert break_even_pd(lgd, POLICY) == pytest.approx(rate / (lgd * aversion + rate))


def test_break_even_pd_falls_as_lgd_rises():
    """Worse recovery prospects mean less PD tolerance."""
    thresholds = [break_even_pd(lgd, POLICY) for lgd in (0.1, 0.25, 0.5, 0.75, 1.0)]
    assert thresholds == sorted(thresholds, reverse=True)


def test_at_break_even_expected_profit_is_approximately_zero():
    """The economic definition of the cut-off."""
    lgd, ead = 0.25, 200_000.0
    result = decide(break_even_pd(lgd, POLICY), lgd, ead, POLICY)
    assert result.expected_profit == pytest.approx(0.0, abs=1.0)


def test_low_risk_application_is_approved():
    result = decide(0.01, 0.25, 200_000.0, POLICY)
    assert result.decision is Decision.APPROVE
    assert result.expected_profit > 0


def test_high_risk_application_is_declined():
    result = decide(0.95, 0.25, 200_000.0, POLICY)
    assert result.decision is Decision.DECLINE


def test_borderline_application_is_reviewed():
    """Within the review band around break-even, a human decides."""
    lgd = 0.25
    result = decide(break_even_pd(lgd, POLICY), lgd, 200_000.0, POLICY)
    assert result.decision is Decision.REVIEW


def test_worst_grade_is_declined_regardless_of_economics():
    """Hard policy limit overrides the economics."""
    result = decide(0.99, 0.10, 200_000.0, POLICY)
    assert result.decision is Decision.DECLINE
    assert "grade" in result.reason.lower()


def test_decision_reason_is_populated():
    assert len(decide(0.05, 0.25, 200_000.0, POLICY).reason) > 20


# ------------------------------------------------------------- concentration


def test_hhi_of_perfectly_concentrated_book_is_one():
    import pandas as pd

    assert herfindahl_index(pd.Series([100.0])) == pytest.approx(1.0)


def test_hhi_of_evenly_spread_book_is_one_over_n():
    import pandas as pd

    assert herfindahl_index(pd.Series([25.0] * 4)) == pytest.approx(0.25)


def test_hhi_rises_with_concentration():
    import pandas as pd

    even = herfindahl_index(pd.Series([25.0, 25.0, 25.0, 25.0]))
    skewed = herfindahl_index(pd.Series([90.0, 5.0, 3.0, 2.0]))
    assert skewed > even
