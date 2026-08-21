"""Drift detection: PSI, KS and calibration monitoring.

HONESTY NOTE, which is repeated in the README and the model card:

This dataset has NO time dimension - ``year`` is 2019 for all 148,670 rows.
There is therefore no real production timeline to monitor and no genuine drift to
observe. Fabricating a timeline would be dishonest.

What this module does instead is implement the detectors properly and
demonstrate them against a *deliberately perturbed* sample, clearly labelled as
a simulation. The detectors are production-shaped: point them at a real scoring
window and they work unchanged.

What degrades first in a production PD model is calibration, not
discrimination - so observed-vs-expected default rate by grade is monitored
alongside the feature-level PSI.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Conventional PSI interpretation bands used across retail credit risk.
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25


@dataclass
class DriftResult:
    feature: str
    psi: float
    severity: str  # "stable" | "moderate" | "significant"
    kind: str  # "numeric" | "categorical"
    detail: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _severity(psi: float) -> str:
    if psi < PSI_NO_SHIFT:
        return "stable"
    if psi < PSI_MODERATE_SHIFT:
        return "moderate"
    return "significant"


def population_stability_index(
    reference: pd.Series,
    current: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> tuple[float, list[dict[str, Any]]]:
    """PSI between a reference and a current distribution.

        PSI = sum( (current% - reference%) * ln(current% / reference%) )

    Numeric series are bucketed on reference quantiles so the bins reflect the
    distribution the model was trained on. Categorical series compare level
    frequencies directly.
    """
    ref = reference.dropna()
    cur = current.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0, []

    is_numeric = pd.api.types.is_numeric_dtype(ref) and ref.nunique() > bins

    if is_numeric:
        quantiles = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        quantiles[0], quantiles[-1] = -np.inf, np.inf
        # Valid at runtime; the stub overloads do not cover a Series with
        # ndarray bin edges.
        ref_counts = pd.cut(ref, quantiles).value_counts().sort_index()  # type: ignore[call-overload]
        cur_counts = pd.cut(cur, quantiles).value_counts().sort_index()  # type: ignore[call-overload]
    else:
        levels = sorted(set(ref.unique()) | set(cur.unique()), key=str)
        ref_counts = ref.value_counts().reindex(levels, fill_value=0)
        cur_counts = cur.value_counts().reindex(levels, fill_value=0)

    ref_pct = (ref_counts / ref_counts.sum()).clip(lower=epsilon)
    cur_pct = (cur_counts / cur_counts.sum()).clip(lower=epsilon)
    contributions = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)

    detail = [
        {
            "bucket": str(bucket),
            "reference_pct": float(ref_pct.loc[bucket]),
            "current_pct": float(cur_pct.loc[bucket]),
            "contribution": float(contributions.loc[bucket]),
        }
        for bucket in ref_pct.index
    ]
    return float(contributions.sum()), detail


def feature_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str] | None = None,
    bins: int = 10,
) -> list[DriftResult]:
    """PSI for every shared column, worst first."""
    columns = columns or [c for c in reference.columns if c in current.columns]
    results = []
    for column in columns:
        psi, detail = population_stability_index(reference[column], current[column], bins)
        results.append(
            DriftResult(
                feature=column,
                psi=psi,
                severity=_severity(psi),
                kind="numeric"
                if pd.api.types.is_numeric_dtype(reference[column])
                else "categorical",
                detail=detail,
            )
        )
    return sorted(results, key=lambda r: r.psi, reverse=True)


def prediction_drift(reference_pd, current_pd, bins: int = 10) -> DriftResult:
    """PSI on the predicted PD distribution itself.

    The single most operationally useful drift metric: it moves whenever the
    input mix changes in any way the model responds to, without needing labels.
    """
    psi, detail = population_stability_index(
        pd.Series(np.asarray(reference_pd, dtype=float)),
        pd.Series(np.asarray(current_pd, dtype=float)),
        bins,
    )
    return DriftResult(
        feature="predicted_pd", psi=psi, severity=_severity(psi), kind="numeric", detail=detail
    )


def calibration_drift(y_true, y_prob, n_bins: int = 10) -> dict[str, Any]:
    """Observed vs expected default rate by PD decile.

    Calibration is what degrades first in production. A model can keep its
    ranking power (stable AUC) while its absolute PD estimates drift badly - and
    it is the absolute PD that feeds expected loss and pricing.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, dtype=float)}
    )
    try:
        frame["bin"] = pd.qcut(frame["p"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        frame["bin"] = 0

    grouped = frame.groupby("bin", observed=True).agg(
        n=("y", "size"), expected=("p", "mean"), observed=("y", "mean")
    )
    grouped["gap"] = grouped["observed"] - grouped["expected"]

    total_expected = float(frame["p"].mean())
    total_observed = float(frame["y"].mean())
    return {
        "expected_default_rate": total_expected,
        "observed_default_rate": total_observed,
        "overall_gap": total_observed - total_expected,
        "mean_absolute_gap": float(grouped["gap"].abs().mean()),
        "max_absolute_gap": float(grouped["gap"].abs().max()),
        "bins": grouped.reset_index().to_dict(orient="records"),
        "alert": bool(abs(total_observed - total_expected) > 0.02),
    }


def data_quality_report(df: pd.DataFrame, contract: dict[str, Any]) -> dict[str, Any]:
    """Null rates, out-of-range values and unseen categories in a scoring batch.

    Reuses the same contract that drives request validation, so monitoring and
    validation cannot disagree about what "valid" means.
    """
    issues: list[dict[str, Any]] = []

    for column, spec in contract.get("numeric", {}).items():
        if column not in df.columns:
            continue
        series = df[column]
        out_of_range = int(((series < spec["min"]) | (series > spec["max"])).sum())
        if out_of_range:
            issues.append({"column": column, "issue": "out_of_range", "count": out_of_range})
        if (nulls := int(series.isna().sum())) and not spec.get("nullable", False):
            issues.append({"column": column, "issue": "unexpected_null", "count": nulls})

    for column, spec in contract.get("categorical", {}).items():
        if column not in df.columns:
            continue
        allowed = set(spec["allowed"])
        unseen = set(df[column].dropna().unique()) - allowed
        if unseen:
            issues.append(
                {"column": column, "issue": "unseen_category", "values": sorted(map(str, unseen))}
            )

    return {"n_rows": int(len(df)), "n_issues": len(issues), "issues": issues, "passed": not issues}
