"""PD to risk grade mapping.

Grades are a *policy* construct, not a model output. The boundaries live in
config/risk_policy.yaml and are calibrated to this portfolio's PD distribution;
they are not comparable to any external master scale (Moody's, S&P, or an
internal bank scale).

Serving a grade rather than a raw probability is how lending decisions are
actually communicated and governed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from credit_risk.config import load_risk_policy


@dataclass(frozen=True)
class Grade:
    grade: str
    max_pd: float
    description: str


def grade_scale(policy: dict | None = None) -> list[Grade]:
    """The ordered master scale, best grade first."""
    cfg = policy or load_risk_policy()
    scale = [Grade(**g) for g in cfg["grades"]]
    return sorted(scale, key=lambda g: g.max_pd)


def assign_grade(pd_value: float, policy: dict | None = None) -> str:
    """Map a single calibrated PD to its grade."""
    for grade in grade_scale(policy):
        if pd_value < grade.max_pd:
            return grade.grade
    return grade_scale(policy)[-1].grade


def assign_grades(pd_values, policy: dict | None = None) -> np.ndarray:
    """Vectorised grade assignment."""
    scale = grade_scale(policy)
    bounds = [g.max_pd for g in scale]
    labels = [g.grade for g in scale]
    idx = np.searchsorted(bounds, np.asarray(pd_values, dtype=float), side="right")
    idx = np.clip(idx, 0, len(labels) - 1)
    return np.asarray(labels, dtype=object)[idx]


def grade_summary(pd_values, y_true=None, policy: dict | None = None) -> pd.DataFrame:
    """Distribution across grades, with observed default rate where labels exist.

    Used to demonstrate rank-ordering: the observed default rate must increase
    monotonically from grade A through G. ``tests/model/test_grade_monotonicity.py``
    asserts this.
    """
    scale = grade_scale(policy)
    order = [g.grade for g in scale]
    df = pd.DataFrame(
        {"pd": np.asarray(pd_values, dtype=float), "grade": assign_grades(pd_values, policy)}
    )
    if y_true is not None:
        df["y"] = np.asarray(y_true).astype(int)

    agg: dict = {
        "n": ("pd", "size"),
        "mean_pd": ("pd", "mean"),
        "min_pd": ("pd", "min"),
        "max_pd": ("pd", "max"),
    }
    if y_true is not None:
        agg["observed_default_rate"] = ("y", "mean")

    out = df.groupby("grade", observed=True).agg(**agg)
    out = out.reindex(order).dropna(subset=["n"])
    out["share"] = out["n"] / out["n"].sum()
    return out.reset_index()


def is_monotonic(summary: pd.DataFrame, column: str = "observed_default_rate") -> bool:
    """True when the observed default rate is non-decreasing across grades."""
    if column not in summary.columns:
        raise KeyError(f"{column!r} not in summary; pass y_true to grade_summary()")
    values = summary[column].to_numpy(dtype=float)
    return bool(np.all(np.diff(values) >= 0))
