"""SHAP-based prediction explanations and reason codes.

Reason codes are a functional requirement rather than a presentation nicety.
Lenders are generally expected to tell a declined applicant the specific
principal reasons for that decision, so a scoring model that can only emit a
number is not much use at the point where it matters most.

Honesty note that is surfaced in the API response: SHAP is computed on the
UNCALIBRATED model score. Isotonic calibration is monotonic, so the sign and the
ranking of contributions carry over to the calibrated PD, but the magnitudes are
in the log-odds space of the uncalibrated model and must not be read as
"this feature added N percentage points of PD".
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

# Human-readable names for the reason codes. Anything not listed falls back to a
# prettified version of the raw column name.
FEATURE_LABELS: dict[str, str] = {
    "loan_to_income": "Loan-to-income ratio",
    "property_to_income": "Property value relative to income",
    "loan_to_value_ratio": "Loan amount relative to collateral",
    "payment_to_income": "Monthly payment burden",
    "high_dti": "Debt-to-income above the 43% QM threshold",
    "dtir1": "Debt-to-income ratio",
    "LTV": "Loan-to-value ratio",
    "loan_amount": "Loan amount",
    "property_value": "Property value",
    "income": "Applicant income",
    "term": "Loan term",
    "Credit_Score": "Credit score",
    "lump_sum_payment": "Balloon / lump-sum repayment structure",
    "Neg_ammortization": "Negative amortisation feature",
    "interest_only": "Interest-only repayment",
    "business_or_commercial": "Business or commercial purpose",
    "occupancy_type": "Occupancy type",
    "loan_purpose": "Loan purpose",
    "credit_type": "Credit bureau used",
    "co-applicant_credit_type": "Co-applicant credit bureau",
    "submission_of_application": "Application channel",
    "Region": "Region",
    "loan_limit": "Conforming loan limit status",
    "approv_in_adv": "Pre-approval status",
    "loan_type": "Loan type",
    "Credit_Worthiness": "Creditworthiness category",
    "open_credit": "Open credit lines",
    "construction_type": "Construction type",
    "Secured_by": "Collateral type",
    "total_units": "Number of units",
    "Security_Type": "Security type",
}


@dataclass
class ReasonCode:
    feature: str
    label: str
    contribution: float  # signed SHAP contribution, log-odds space
    direction: str  # "increases_risk" | "reduces_risk"
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Explanation:
    risk_drivers: list[ReasonCode]
    risk_reducers: list[ReasonCode]
    base_value: float
    method: str = "shap_tree_explainer"
    note: str = (
        "SHAP values are computed on the uncalibrated model score in log-odds "
        "space. Calibration is monotonic, so signs and rankings carry over to the "
        "calibrated PD, but magnitudes are not percentage-point contributions."
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "risk_drivers": [r.to_dict() for r in self.risk_drivers],
            "risk_reducers": [r.to_dict() for r in self.risk_reducers],
            "base_value": self.base_value,
            "method": self.method,
            "note": self.note,
        }


def unwrap_pipeline(model: Any) -> Pipeline:
    """Recover the inner sklearn Pipeline from a calibrated wrapper."""
    if isinstance(model, Pipeline):
        return model
    if isinstance(model, CalibratedClassifierCV):
        inner = model.calibrated_classifiers_[0].estimator
        # FrozenEstimator wraps the real estimator in `.estimator`
        inner = getattr(inner, "estimator", inner)
        return unwrap_pipeline(inner)
    if hasattr(model, "estimator"):
        return unwrap_pipeline(model.estimator)
    raise TypeError(f"Cannot locate a Pipeline inside {type(model).__name__}")


def _source_column(encoded_name: str, source_columns: list[str]) -> str:
    """Map an encoded feature name back to the raw column it came from.

    ColumnTransformer emits names like ``num__loan_amount`` or
    ``cat__Region_North``. One-hot columns must be folded back to ``Region`` so
    contributions aggregate to one reason code per underlying feature.
    """
    name = re.sub(r"^(num|cat|bin|remainder)__", "", encoded_name)
    if name in source_columns:
        return name
    # One-hot: longest matching prefix wins, so `co-applicant_credit_type_CIB`
    # does not get attributed to `credit_type`.
    matches = [c for c in source_columns if name.startswith(f"{c}_")]
    return max(matches, key=len) if matches else name


class PredictionExplainer:
    """Produces reason codes for individual predictions.

    The TreeExplainer is constructed once at startup and reused; building it per
    request would dominate inference latency.
    """

    def __init__(self, model: Any, max_reasons: int = 4):
        self.pipeline = unwrap_pipeline(model)
        self.max_reasons = max_reasons
        self._transform = Pipeline(self.pipeline.steps[:-1])
        self._estimator = self.pipeline.steps[-1][1]
        self._encoded_names = list(self.pipeline.named_steps["preprocess"].get_feature_names_out())
        try:
            self._explainer = shap.TreeExplainer(self._estimator)
            self.available = True
        except Exception as exc:  # non-tree model, e.g. logistic regression
            logger.warning(
                "TreeExplainer unavailable for %s: %s", type(self._estimator).__name__, exc
            )
            self._explainer = None
            self.available = False

    def explain(self, X: pd.DataFrame, source_columns: list[str] | None = None) -> Explanation:
        """Explain a single-row frame."""
        if not self.available:
            return Explanation(
                risk_drivers=[], risk_reducers=[], base_value=0.0, method="unavailable"
            )

        source_columns = source_columns or list(X.columns)
        transformed = self._transform.transform(X)
        shap_values = self._explainer.shap_values(transformed)
        values = np.asarray(shap_values)
        if values.ndim == 3:  # (n, features, classes)
            values = values[..., -1]
        row = values[0]

        base = self._explainer.expected_value
        base = float(np.asarray(base).ravel()[-1])

        # Aggregate encoded contributions back to source features.
        contributions: dict[str, float] = {}
        for encoded, contribution in zip(self._encoded_names, row, strict=True):
            key = _source_column(encoded, source_columns)
            contributions[key] = contributions.get(key, 0.0) + float(contribution)

        raw_values = X.iloc[0].to_dict()
        ranked = sorted(contributions.items(), key=lambda kv: abs(kv[1]), reverse=True)

        drivers: list[ReasonCode] = []
        reducers: list[ReasonCode] = []
        for feature, contribution in ranked:
            if abs(contribution) < 1e-6:
                continue
            code = ReasonCode(
                feature=feature,
                label=FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize()),
                contribution=round(float(contribution), 6),
                direction="increases_risk" if contribution > 0 else "reduces_risk",
                value=_clean(raw_values.get(feature)),
            )
            (drivers if contribution > 0 else reducers).append(code)

        return Explanation(
            risk_drivers=drivers[: self.max_reasons],
            risk_reducers=reducers[: self.max_reasons],
            base_value=base,
        )

    def global_importance(self, X: pd.DataFrame, sample: int = 2000) -> pd.DataFrame:
        """Mean |SHAP| per source feature - the global importance ranking."""
        if not self.available:
            return pd.DataFrame(columns=["feature", "mean_abs_shap"])
        subset = X.sample(n=min(sample, len(X)), random_state=42) if len(X) > sample else X
        transformed = self._transform.transform(subset)
        values = np.asarray(self._explainer.shap_values(transformed))
        if values.ndim == 3:
            values = values[..., -1]

        source_columns = list(X.columns)
        totals: dict[str, float] = {}
        for encoded, column in zip(self._encoded_names, np.abs(values).mean(axis=0), strict=True):
            key = _source_column(encoded, source_columns)
            totals[key] = totals.get(key, 0.0) + float(column)

        return (
            pd.DataFrame({"feature": list(totals), "mean_abs_shap": list(totals.values())})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )


def _clean(value: Any) -> Any:
    """Make a raw feature value JSON-serialisable."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
