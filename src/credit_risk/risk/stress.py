"""Portfolio stress and sensitivity analysis.

WHAT THIS IS: input shocks applied to borrower and collateral variables, re-scored
through the actual PD model, aggregated to portfolio expected loss.

WHAT THIS IS NOT: a CCAR or DFAST exercise. Those apply supervisor-prescribed
macroeconomic paths to models estimated on multi-year panel data. This dataset
has no macroeconomic variables and no time dimension at all (``year`` == 2019 for
every one of the 148,670 rows), so there is no basis for conditioning PD on a
macro scenario. Calling this "CCAR-compliant" would be fiction.

Honest limitation, stated on every result: the model was fitted on a single
cross-section, so its response to a 25% income shock is an extrapolation beyond
the training distribution. Results indicate direction and rough magnitude only.

Note also that interest-rate shocks are NOT available. ``rate_of_interest`` and
``Interest_rate_spread`` are excluded from the model as target leakage, so the
model has no rate sensitivity to stress.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from credit_risk.config import load_risk_policy, load_stress_scenarios
from credit_risk.risk.portfolio import PortfolioSummary, aggregate

logger = logging.getLogger(__name__)

# Only these may be shocked. Anything else is either excluded from the model or
# not economically meaningful to shock.
SHOCKABLE = {"income", "property_value", "dtir1", "loan_amount", "Credit_Score"}


@dataclass
class ScenarioResult:
    name: str
    label: str
    description: str
    shocks: dict[str, float]
    summary: PortfolioSummary
    # Deltas against base
    delta_expected_loss: float = 0.0
    delta_expected_loss_pct: float = 0.0
    delta_weighted_pd: float = 0.0
    grade_migration: dict[str, int] = field(default_factory=dict)
    n_underwater: int = 0
    extrapolation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["summary"] = self.summary.to_dict()
        return out


def extrapolation_diagnostics(
    reference: pd.DataFrame,
    stressed: pd.DataFrame,
    columns: tuple[str, ...] = ("LTV", "dtir1", "loan_to_income"),
) -> dict[str, Any]:
    """How far outside the observed distribution has this scenario pushed the book?

    This is the honest counterweight to a dramatic stress number. The model
    learned the high-LTV region from very few observations - in the training data
    only ~1.2% of loans sit above LTV 100 - so a large collateral shock moves a
    substantial share of the portfolio into territory the model has barely seen.
    Its response there is an extrapolation, and the reported loss should be read
    with that in mind rather than as a forecast.

    For each column we report the share of stressed rows beyond the 99th
    percentile of the unstressed portfolio, which stands in for the training
    envelope (the portfolio is drawn from the same population).
    """
    diagnostics: dict[str, Any] = {}
    for column in columns:
        if column not in reference.columns or column not in stressed.columns:
            continue
        envelope = float(reference[column].quantile(0.99))
        beyond = float((stressed[column] > envelope).mean())
        diagnostics[column] = {
            "reference_p99": envelope,
            "share_beyond_reference_p99": beyond,
            "stressed_median": float(stressed[column].median()),
        }

    worst = max((d["share_beyond_reference_p99"] for d in diagnostics.values()), default=0.0)
    diagnostics["max_share_outside_envelope"] = worst
    diagnostics["confidence"] = "low" if worst > 0.20 else "moderate" if worst > 0.05 else "high"
    diagnostics["note"] = (
        "Share of the stressed book beyond the 99th percentile of the unstressed "
        "portfolio. A large share means the model is extrapolating and the loss "
        "estimate indicates direction rather than magnitude."
    )
    return diagnostics


def apply_shocks(
    df: pd.DataFrame,
    shocks: dict[str, float],
    recompute_ltv: bool = True,
) -> pd.DataFrame:
    """Apply relative shocks to borrower/collateral inputs.

    Shocks are relative (``-0.25`` means a 25% reduction). When
    ``recompute_ltv`` is set, LTV is recalculated from the shocked property value
    rather than shocked independently, so the collateral channel stays
    internally consistent: a property shock raises LTV (a model input, so PD
    rises) *and* reduces recoverable collateral (so the LGD proxy rises too).
    That joint movement is the mechanism that matters in a mortgage book.
    """
    out = df.copy()
    for column, magnitude in shocks.items():
        if column not in SHOCKABLE:
            raise ValueError(
                f"{column!r} is not shockable. Allowed: {sorted(SHOCKABLE)}. "
                "Interest rate variables are excluded from the model as leakage."
            )
        if column not in out.columns:
            logger.warning("shock column %s not present, skipped", column)
            continue
        out[column] = out[column] * (1.0 + float(magnitude))

    if recompute_ltv and "property_value" in shocks and "LTV" in out.columns:
        original_pv = df["property_value"].to_numpy(dtype=float)
        shocked_pv = out["property_value"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            factor = np.divide(
                original_pv, shocked_pv, out=np.ones_like(original_pv), where=shocked_pv > 0
            )
        out["LTV"] = df["LTV"].to_numpy(dtype=float) * factor

    return out


def run_scenario(
    model,
    df: pd.DataFrame,
    scenario: dict[str, Any],
    *,
    recompute_ltv: bool = True,
    policy: dict | None = None,
    segments: pd.DataFrame | None = None,
) -> ScenarioResult:
    """Shock, re-score through the model, aggregate."""
    cfg = policy or load_risk_policy()
    shocks = scenario.get("shocks") or {}
    stressed = apply_shocks(df, shocks, recompute_ltv) if shocks else df.copy()

    pd_values = model.predict_proba(stressed)[:, 1]
    summary = aggregate(
        pd_values,
        stressed["loan_amount"],
        stressed.get("property_value"),
        segments=segments,
        policy=cfg,
    )

    underwater = 0
    if "LTV" in stressed.columns:
        underwater = int((stressed["LTV"] > 100).sum())

    return ScenarioResult(
        name=scenario["name"],
        label=scenario.get("label", scenario["name"]),
        description=scenario.get("description", "").strip(),
        shocks=shocks,
        summary=summary,
        n_underwater=underwater,
        extrapolation=extrapolation_diagnostics(df, stressed),
    )


def run_all_scenarios(
    model,
    df: pd.DataFrame,
    scenarios_cfg: dict[str, Any] | None = None,
    policy: dict | None = None,
    segments: pd.DataFrame | None = None,
) -> list[ScenarioResult]:
    """Run every configured scenario and compute deltas against the base case."""
    cfg = scenarios_cfg or load_stress_scenarios()
    recompute = bool(cfg.get("recompute_ltv_from_property_value", True))

    results = [
        run_scenario(model, df, sc, recompute_ltv=recompute, policy=policy, segments=segments)
        for sc in cfg["scenarios"]
    ]

    base = next((r for r in results if r.name == "base"), results[0])
    base_el = base.summary.total_expected_loss
    base_grades = {g["grade"]: g["n"] for g in base.summary.grade_distribution}

    for result in results:
        result.delta_expected_loss = result.summary.total_expected_loss - base_el
        result.delta_expected_loss_pct = result.delta_expected_loss / base_el if base_el else 0.0
        result.delta_weighted_pd = (
            result.summary.weighted_average_pd - base.summary.weighted_average_pd
        )
        current = {g["grade"]: g["n"] for g in result.summary.grade_distribution}
        all_grades = sorted(set(base_grades) | set(current))
        result.grade_migration = {
            g: int(current.get(g, 0) - base_grades.get(g, 0)) for g in all_grades
        }

    return results


def sensitivity_sweep(
    model,
    df: pd.DataFrame,
    variable: str,
    magnitudes: list[float],
    policy: dict | None = None,
) -> pd.DataFrame:
    """Single-variable sweep, for tornado charts.

    Isolates each driver's contribution, which is what distinguishes sensitivity
    analysis from scenario analysis.
    """
    rows = []
    for magnitude in magnitudes:
        result = run_scenario(
            model,
            df,
            {"name": f"{variable}{magnitude:+.0%}", "shocks": {variable: magnitude}},
            policy=policy,
        )
        rows.append(
            {
                "variable": variable,
                "shock": magnitude,
                "weighted_average_pd": result.summary.weighted_average_pd,
                "expected_loss": result.summary.total_expected_loss,
                "expected_loss_rate": result.summary.expected_loss_rate,
            }
        )
    return pd.DataFrame(rows)


LIMITATIONS = [
    "Assumption-driven sensitivity analysis, not a macro-conditioned forecast. Not CCAR or DFAST.",
    "The dataset has no time dimension (year == 2019 for all rows) and no "
    "macroeconomic variables, so PD cannot be conditioned on a macro path.",
    "The model was fitted on a single cross-section; response to large shocks is "
    "an extrapolation beyond the training distribution.",
    "Interest-rate shocks are unavailable: rate variables are excluded from the "
    "model as target leakage, so the model has no rate sensitivity.",
]
