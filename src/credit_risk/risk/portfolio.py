"""Portfolio-level risk aggregation.

Single-loan decisioning answers "should we lend to this borrower?". Portfolio
aggregation answers "what do we hold, and where is it concentrated?" - which is
the question a credit risk function actually reports on.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import load_risk_policy
from credit_risk.risk.expected_loss import expected_loss
from credit_risk.risk.grades import assign_grades, grade_scale


@dataclass
class PortfolioSummary:
    n_exposures: int
    total_exposure: float
    total_expected_loss: float
    expected_loss_rate: float  # EL / EAD, in bps terms when x10000
    weighted_average_pd: float
    weighted_average_lgd: float
    grade_distribution: list[dict[str, Any]] = field(default_factory=list)
    concentrations: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"  Exposures            {self.n_exposures:,}\n"
            f"  Total exposure       {self.total_exposure:,.0f}\n"
            f"  Expected loss        {self.total_expected_loss:,.0f}\n"
            f"  EL rate              {self.expected_loss_rate:.4%} "
            f"({self.expected_loss_rate * 10000:.0f} bps)\n"
            f"  Exposure-wtd PD      {self.weighted_average_pd:.4%}\n"
            f"  Exposure-wtd LGD     {self.weighted_average_lgd:.4%}"
        )


def herfindahl_index(exposures: pd.Series) -> float:
    """Herfindahl-Hirschman Index of exposure concentration.

    Sum of squared exposure shares. 1/n means perfectly diversified across n
    buckets; 1.0 means everything sits in one bucket. Reported per segment so a
    reviewer can see whether the book is concentrated in a region or purpose.
    """
    total = exposures.sum()
    if total <= 0:
        return 0.0
    shares = exposures / total
    return float((shares**2).sum())


def concentration_report(
    df: pd.DataFrame,
    exposure_column: str = "ead",
    el_column: str = "expected_loss",
    segment_columns: tuple[str, ...] = ("Region", "loan_purpose", "occupancy_type"),
) -> dict[str, Any]:
    """Exposure and expected loss by segment, with an HHI per dimension."""
    report: dict[str, Any] = {}
    total_exposure = float(df[exposure_column].sum())

    for col in segment_columns:
        if col not in df.columns:
            continue
        grouped = df.groupby(col, observed=True).agg(
            n=(exposure_column, "size"),
            exposure=(exposure_column, "sum"),
            expected_loss=(el_column, "sum"),
        )
        grouped["exposure_share"] = grouped["exposure"] / max(total_exposure, 1e-9)
        grouped["el_rate"] = grouped["expected_loss"] / grouped["exposure"].replace(0, np.nan)
        report[col] = {
            "hhi": herfindahl_index(grouped["exposure"]),
            "effective_buckets": (
                1.0 / herfindahl_index(grouped["exposure"])
                if herfindahl_index(grouped["exposure"]) > 0
                else 0.0
            ),
            "segments": grouped.reset_index().to_dict(orient="records"),
        }
    return report


def aggregate(
    pd_values,
    loan_amount,
    property_value=None,
    segments: pd.DataFrame | None = None,
    policy: dict | None = None,
) -> PortfolioSummary:
    """Aggregate a scored book into portfolio metrics.

    ``segments`` is an optional frame of categorical columns (Region, purpose,
    ...) aligned with the exposures, used for the concentration report.
    """
    cfg = policy or load_risk_policy()
    components = expected_loss(pd_values, loan_amount, property_value, cfg)

    ead = components["ead"]
    el = components["expected_loss"]
    total_ead = float(ead.sum())

    frame = pd.DataFrame(
        {
            "pd": components["pd"],
            "lgd": components["lgd"],
            "ead": ead,
            "expected_loss": el,
            "grade": assign_grades(components["pd"], cfg),
        }
    )
    if segments is not None:
        for col in segments.columns:
            frame[col] = np.asarray(segments[col])

    order = [g.grade for g in grade_scale(cfg)]
    by_grade = (
        frame.groupby("grade", observed=True)
        .agg(
            n=("pd", "size"),
            exposure=("ead", "sum"),
            expected_loss=("expected_loss", "sum"),
            mean_pd=("pd", "mean"),
            mean_lgd=("lgd", "mean"),
        )
        .reindex(order)
        .dropna(subset=["n"])
    )
    by_grade["exposure_share"] = by_grade["exposure"] / max(total_ead, 1e-9)

    return PortfolioSummary(
        n_exposures=int(len(frame)),
        total_exposure=total_ead,
        total_expected_loss=float(el.sum()),
        expected_loss_rate=float(el.sum() / total_ead) if total_ead else 0.0,
        weighted_average_pd=float(np.average(components["pd"], weights=ead)) if total_ead else 0.0,
        weighted_average_lgd=float(np.average(components["lgd"], weights=ead))
        if total_ead
        else 0.0,
        grade_distribution=by_grade.reset_index().to_dict(orient="records"),
        concentrations=concentration_report(frame) if segments is not None else {},
    )
