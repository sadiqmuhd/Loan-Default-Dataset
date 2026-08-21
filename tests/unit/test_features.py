"""Feature engineering unit tests, including the edge cases present in the data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_risk.features.engineering import add_features

BASE = {
    "loan_amount": 296_500.0,
    "income": 5_760.0,
    "property_value": 418_000.0,
    "term": 360.0,
    "dtir1": 39.0,
}


def frame(**overrides) -> pd.DataFrame:
    return pd.DataFrame([{**BASE, **overrides}])


def test_loan_to_income_annualises_monthly_income():
    """income is MONTHLY in this dataset; LTI must annualise it.

    The original code computed loan_amount / income directly, giving a median
    loan-to-income of 52x - not a real mortgage. Annualised it is 4.36x.
    """
    result = add_features(frame())
    expected = 296_500.0 / (5_760.0 * 12)
    assert result["loan_to_income"].iloc[0] == pytest.approx(expected)
    assert 3.0 < result["loan_to_income"].iloc[0] < 6.0


def test_add_features_does_not_mutate_input():
    original = frame()
    before = original.copy()
    add_features(original)
    pd.testing.assert_frame_equal(original, before)


@pytest.mark.parametrize("income", [0.0, -100.0, -0.000001])
def test_non_positive_income_yields_nan_not_infinity(income):
    """1,260 real rows have income <= 0.

    The original code only guarded against exactly zero via .replace(0, np.nan),
    so negative incomes produced negative ratios and were fed to the model.
    """
    result = add_features(frame(income=income))
    assert np.isnan(result["loan_to_income"].iloc[0])
    assert np.isnan(result["property_to_income"].iloc[0])
    assert not np.isinf(result["loan_to_income"].iloc[0])


def test_zero_property_value_yields_nan():
    result = add_features(frame(property_value=0.0))
    assert np.isnan(result["loan_to_value_ratio"].iloc[0])


def test_zero_term_yields_nan_payment_burden():
    result = add_features(frame(term=0.0))
    assert np.isnan(result["payment_to_income"].iloc[0])


@pytest.mark.parametrize(
    ("dtir1", "expected"),
    [(42.9, 0.0), (43.0, 0.0), (43.1, 1.0), (60.0, 1.0), (0.0, 0.0)],
)
def test_high_dti_uses_qm_threshold(dtir1, expected):
    """43% is the Ability-to-Repay / QM threshold, 12 CFR 1026.43(e).
    The boundary is exclusive: exactly 43.0 is not 'high'."""
    result = add_features(frame(dtir1=dtir1))
    assert result["high_dti"].iloc[0] == expected


def test_high_dti_is_nan_when_dti_unknown():
    """An unknown DTI must not be silently coded as 'not high'.

    The original ``(df['dtir1'] > 50).astype(int)`` mapped NaN to 0, asserting
    that a borrower with unknown debt burden is low risk.
    """
    result = add_features(frame(dtir1=np.nan))
    assert np.isnan(result["high_dti"].iloc[0])


def test_loan_to_value_ratio_is_a_percentage():
    result = add_features(frame(loan_amount=209_000.0, property_value=418_000.0))
    assert result["loan_to_value_ratio"].iloc[0] == pytest.approx(50.0)


def test_all_engineered_features_present(model_config):
    result = add_features(frame())
    for feature in model_config["features"]["engineered"]:
        assert feature in result.columns


def test_handles_multiple_rows():
    df = pd.concat([frame(), frame(income=0.0), frame(dtir1=np.nan)], ignore_index=True)
    result = add_features(df)
    assert len(result) == 3
    assert not np.isnan(result["loan_to_income"].iloc[0])
    assert np.isnan(result["loan_to_income"].iloc[1])
    assert np.isnan(result["high_dti"].iloc[2])
