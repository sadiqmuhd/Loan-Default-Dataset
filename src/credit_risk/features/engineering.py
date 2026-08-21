"""Feature engineering. One implementation, imported by both training and serving.

The original codebase had two copies of this logic - one in ``modeltraining.py``
and one in ``app/feature_engineering.py`` - which is how training and serving
silently drift apart. This module is the only definition, and
``tests/model/test_train_serve_parity.py`` asserts both paths produce identical
output.

Deliberately absent: missingness indicators. In the original code
``add_features`` created ``Interest_rate_spread_missing``, which was measured to
equal the target for 148,670 of 148,670 rows. Missingness indicators are
forbidden here; ``tests/model/test_no_leakage.py`` enforces it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from credit_risk.config import load_model_config

# Guard against divide-by-zero producing inf; these are ratios, so a missing
# denominator must become NaN and be imputed, never infinity.
_EPS = 1e-9


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise ratio where a non-positive denominator yields NaN."""
    denom = denominator.where(denominator > _EPS)
    return numerator / denom


def add_features(df: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Add engineered credit features to a raw loan frame.

    Pure function: does not mutate the input.

    Parameters
    ----------
    df
        Raw loan records containing at least ``loan_amount``, ``income``,
        ``term``, ``property_value`` and ``dtir1``.
    params
        Feature parameters. Defaults to ``feature_params`` in config/model.yaml.
    """
    if params is None:
        params = load_model_config()["feature_params"]

    periods = float(params["income_periods_per_year"])
    dti_threshold = float(params["high_dti_threshold"])

    out = df.copy()
    annual_income = out["income"] * periods

    # Loan-to-income: the standard mortgage affordability metric. `income` is
    # monthly in this dataset, so it must be annualised first.
    out["loan_to_income"] = _safe_ratio(out["loan_amount"], annual_income)
    out["property_to_income"] = _safe_ratio(out["property_value"], annual_income)

    # Loan amount against collateral. Distinct from the supplied `LTV` column,
    # which is reported by the originator; this is computed and lets the model
    # see disagreement between the two.
    out["loan_to_value_ratio"] = 100.0 * _safe_ratio(out["loan_amount"], out["property_value"])

    # Principal-only monthly payment as a share of monthly income. A lower bound
    # on true payment burden, since it excludes interest, taxes and insurance.
    monthly_principal = _safe_ratio(out["loan_amount"], out["term"])
    out["payment_to_income"] = 100.0 * _safe_ratio(monthly_principal, out["income"])

    # Ability-to-Repay / Qualified Mortgage threshold, 12 CFR 1026.43(e).
    # NaN-safe: an unknown DTI must not be silently coded as "not high".
    out["high_dti"] = np.where(
        out["dtir1"].isna(), np.nan, (out["dtir1"] > dti_threshold).astype(float)
    )

    return out


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """sklearn-compatible wrapper so feature engineering lives inside the pipeline.

    Keeping this in the fitted pipeline is what removes the train/serve skew in
    the original code, where the imputer was fitted outside the pipeline during
    training and never applied at inference.
    """

    def __init__(self, params: dict | None = None):
        self.params = params

    def fit(self, X: pd.DataFrame, y=None):  # noqa: N803
        self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:  # noqa: N803
        return add_features(X, self.params)

    def get_feature_names_out(self, input_features=None):
        cfg = load_model_config()
        base = list(input_features) if input_features is not None else self.feature_names_in_
        return np.asarray(base + list(cfg["features"]["engineered"]), dtype=object)
