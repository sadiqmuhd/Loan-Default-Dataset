"""Credit decision policy.

The approve/decline cut-off is derived rather than picked, from two ideas that
have to be kept separate.

BREAK-EVEN is the PD at which expected margin merely offsets expected loss:

    (1 - PD) * EAD * rate  ==  PD * LGD * EAD * loss_aversion
    PD_breakeven = rate / (LGD * loss_aversion + rate)

That is a floor, not a policy. A bank writing loans at break-even earns nothing
on the capital it has tied up, so the actual cut-off is the HURDLE PD: the point
where risk-adjusted return on capital equals the cost of that capital.

    RAROC = (expected_revenue - expected_loss) / (capital_ratio * EAD)
    approve while RAROC >= cost_of_equity

Expanding, with EAD cancelling throughout:

    (1 - PD) * rate - PD * LGD * loss_aversion  ==  cost_of_equity * capital_ratio
    PD_hurdle = (rate - cost_of_equity * capital_ratio) / (rate + LGD * loss_aversion)

HORIZON. `rate` is the margin over `pd_horizon_years`, matching the horizon of
the PD itself. This is load-bearing. An earlier version multiplied the annual
margin by a seven-year expected life and compared it against a twelve-month PD,
which overstated every loan by about 7x and yielded a 58% cut-off - a policy
that would approve a borrower with a coin-flip chance of default inside a year.
Both quantities now come from the same period.

Because the cut-off depends on LGD, a well-collateralised borrower is tolerated
at a higher PD than a thinly-collateralised one. That falls out of the
arithmetic rather than being coded.

WHAT IS NOT MODELLED: no discounting (immaterial over one year, material if the
horizon is ever extended), no term structure of default, no prepayment, and a
flat capital ratio rather than a risk-weighted one. See MODEL_CARD.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from loan_default.config import load_risk_policy
from loan_default.risk.grades import assign_grade


class Decision(StrEnum):
    APPROVE = "APPROVE"
    REVIEW = "REVIEW"  # to a human underwriter
    DECLINE = "DECLINE"


@dataclass
class DecisionResult:
    decision: Decision
    reason: str
    hurdle_pd: float
    break_even_pd: float
    raroc: float
    cost_of_equity: float
    pd: float
    grade: str
    expected_profit: float
    expected_loss: float
    expected_revenue: float
    capital_required: float
    horizon_years: float

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["decision"] = str(self.decision)
        return out


def revenue_rate(policy: dict | None = None) -> float:
    """Net margin as a fraction of exposure, over the PD's own horizon.

    Deliberately NOT the expected-life margin. See the horizon note in the
    module docstring - matching this to the PD horizon is the whole point.
    """
    econ = (policy or load_risk_policy())["economics"]
    return float(econ["annual_net_margin"]) * float(econ["pd_horizon_years"])


def required_return_on_exposure(policy: dict | None = None) -> float:
    """Return the loan must earn per unit of exposure to clear the hurdle.

    Capital consumed is ``capital_ratio * EAD`` and it must earn ``cost_of_equity``
    over the horizon, so per unit of exposure the requirement is their product.
    """
    econ = (policy or load_risk_policy())["economics"]
    horizon = float(econ["pd_horizon_years"])
    return float(econ["capital_ratio"]) * float(econ["cost_of_equity"]) * horizon


def break_even_pd(lgd: float, policy: dict | None = None) -> float:
    """The PD at which expected margin exactly offsets expected loss.

    Reported for transparency, and always looser than :func:`hurdle_pd`. This is
    the floor below which the loan destroys value outright; it is NOT the
    approval cut-off.
    """
    cfg = policy or load_risk_policy()
    rate = revenue_rate(cfg)
    aversion = float(cfg["economics"]["loss_aversion_multiplier"])
    denominator = lgd * aversion + rate
    if denominator <= 0:
        return 0.0
    return float(np.clip(rate / denominator, 0.0, 1.0))


def hurdle_pd(lgd: float, policy: dict | None = None) -> float:
    """The PD at which RAROC equals the cost of equity. This is the cut-off.

    Returns 0.0 when the margin cannot clear the capital charge even at zero
    PD, which correctly means no loan at this LGD is worth writing.
    """
    cfg = policy or load_risk_policy()
    rate = revenue_rate(cfg)
    aversion = float(cfg["economics"]["loss_aversion_multiplier"])
    required = required_return_on_exposure(cfg)

    denominator = rate + lgd * aversion
    if denominator <= 0:
        return 0.0
    return float(np.clip((rate - required) / denominator, 0.0, 1.0))


def raroc(pd_value: float, lgd: float, policy: dict | None = None) -> float:
    """Risk-adjusted return on the capital the loan consumes.

    Independent of EAD, which cancels. Returns NaN if no capital is held, since
    a return on zero capital is undefined rather than infinite.
    """
    cfg = policy or load_risk_policy()
    rate = revenue_rate(cfg)
    aversion = float(cfg["economics"]["loss_aversion_multiplier"])
    capital_ratio = float(cfg["economics"]["capital_ratio"])
    if capital_ratio <= 0:
        return float("nan")

    net = (1.0 - pd_value) * rate - pd_value * lgd * aversion
    return float(net / capital_ratio)


def decide(
    pd_value: float,
    lgd: float,
    ead: float,
    policy: dict | None = None,
) -> DecisionResult:
    """Apply the economic decision policy to a single application.

    The cut-off is the hurdle PD, not break-even. The review band is expressed
    as a fraction OF the hurdle rather than in absolute percentage points: an
    absolute band sized for a 58% threshold swamps a realistic one near 9%.
    """
    cfg = policy or load_risk_policy()
    econ = cfg["economics"]
    threshold = hurdle_pd(lgd, cfg)
    floor_pd = break_even_pd(lgd, cfg)
    band_fraction = float(cfg["decision"]["manual_review_band_fraction"])
    band = threshold * band_fraction
    hard_decline = str(cfg["decision"]["hard_decline_grade"])

    horizon = float(econ["pd_horizon_years"])
    cost_of_equity = float(econ["cost_of_equity"])
    capital_required = float(econ["capital_ratio"]) * ead

    expected_revenue = (1.0 - pd_value) * ead * revenue_rate(cfg)
    el = pd_value * lgd * ead * float(econ["loss_aversion_multiplier"])
    profit = expected_revenue - el
    achieved_raroc = raroc(pd_value, lgd, cfg)
    grade = assign_grade(pd_value, cfg)

    horizon_note = f"over a {horizon:.0f}-year horizon"

    if grade == hard_decline:
        decision = Decision.DECLINE
        reason = f"Risk grade {grade} is outside credit policy regardless of economics."
    elif pd_value <= threshold - band:
        decision = Decision.APPROVE
        reason = (
            f"PD {pd_value:.2%} clears the hurdle PD of {threshold:.2%} {horizon_note}. "
            f"RAROC {achieved_raroc:.1%} exceeds the {cost_of_equity:.1%} cost of equity."
        )
    elif pd_value >= threshold + band:
        decision = Decision.DECLINE
        reason = (
            f"PD {pd_value:.2%} exceeds the hurdle PD of {threshold:.2%} {horizon_note}. "
            f"RAROC {achieved_raroc:.1%} falls short of the {cost_of_equity:.1%} cost of "
            f"equity, so the loan does not earn its capital."
        )
    else:
        decision = Decision.REVIEW
        reason = (
            f"PD {pd_value:.2%} sits within {band_fraction:.0%} of the hurdle PD of "
            f"{threshold:.2%}. RAROC {achieved_raroc:.1%} is too close to the "
            f"{cost_of_equity:.1%} cost of equity to auto-decide; manual underwriting "
            f"required."
        )

    return DecisionResult(
        decision=decision,
        reason=reason,
        hurdle_pd=threshold,
        break_even_pd=floor_pd,
        raroc=achieved_raroc,
        cost_of_equity=cost_of_equity,
        capital_required=float(capital_required),
        horizon_years=horizon,
        pd=float(pd_value),
        grade=grade,
        expected_profit=float(profit),
        expected_loss=float(el),
        expected_revenue=float(expected_revenue),
    )


def decide_batch(pd_values, lgd_values, ead_values, policy: dict | None = None) -> np.ndarray:
    """Vectorised decisions, for portfolio-level analysis."""
    cfg = policy or load_risk_policy()
    return np.array(
        [
            decide(p, lgd, ead, cfg).decision
            for p, lgd, ead in zip(
                np.asarray(pd_values, dtype=float),
                np.asarray(lgd_values, dtype=float),
                np.asarray(ead_values, dtype=float),
                strict=True,
            )
        ],
        dtype=object,
    )


def policy_disclosure(policy: dict | None = None) -> dict[str, Any]:
    """The economic assumptions behind the cut-off, for the API response."""
    cfg = policy or load_risk_policy()
    econ = cfg["economics"]
    return {
        "pd_horizon_years": float(econ["pd_horizon_years"]),
        "annual_net_margin": float(econ["annual_net_margin"]),
        "revenue_rate_over_horizon": revenue_rate(cfg),
        "capital_ratio": float(econ["capital_ratio"]),
        "cost_of_equity": float(econ["cost_of_equity"]),
        "required_return_on_exposure": required_return_on_exposure(cfg),
        "loss_aversion_multiplier": float(econ["loss_aversion_multiplier"]),
        "manual_review_band_fraction": float(cfg["decision"]["manual_review_band_fraction"]),
        "expected_life_years": float(econ["expected_life_years"]),
        "note": (
            "The cut-off is the hurdle PD, where RAROC equals the cost of equity - "
            "not break-even, which only covers expected loss and earns nothing on "
            "capital. It varies with LGD: better-collateralised exposures are "
            "tolerated at a higher PD. PD and margin share the same horizon "
            "(pd_horizon_years); expected_life_years is recorded for reference and "
            "is deliberately not used in the cut-off."
        ),
    }
