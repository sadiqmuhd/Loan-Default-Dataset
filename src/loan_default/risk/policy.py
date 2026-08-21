"""Credit decision policy.

The approve/decline cut-off is derived rather than picked. It is the PD at
which expected margin equals expected loss:

    (1 - PD) * EAD * margin * life  ==  PD * LGD * EAD * loss_aversion

Solving for PD, and noting EAD cancels:

    PD* = revenue_rate / (LGD * loss_aversion + revenue_rate)
    where revenue_rate = annual_net_margin * expected_life_years

Because the cut-off depends on LGD, a well-collateralised borrower is tolerated
at a higher PD than a thinly-collateralised one. That is the economically
correct behaviour and it falls out of the arithmetic rather than being coded.
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
    break_even_pd: float
    pd: float
    grade: str
    expected_profit: float
    expected_loss: float
    expected_revenue: float

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["decision"] = str(self.decision)
        return out


def revenue_rate(policy: dict | None = None) -> float:
    """Lifetime net margin as a fraction of exposure."""
    econ = (policy or load_risk_policy())["economics"]
    return float(econ["annual_net_margin"]) * float(econ["expected_life_years"])


def break_even_pd(lgd: float, policy: dict | None = None) -> float:
    """The PD at which expected margin exactly offsets expected loss."""
    cfg = policy or load_risk_policy()
    rate = revenue_rate(cfg)
    aversion = float(cfg["economics"]["loss_aversion_multiplier"])
    denominator = lgd * aversion + rate
    if denominator <= 0:
        return 0.0
    return float(rate / denominator)


def decide(
    pd_value: float,
    lgd: float,
    ead: float,
    policy: dict | None = None,
) -> DecisionResult:
    """Apply the economic decision policy to a single application."""
    cfg = policy or load_risk_policy()
    threshold = break_even_pd(lgd, cfg)
    band = float(cfg["decision"]["manual_review_margin"])
    hard_decline = str(cfg["decision"]["hard_decline_grade"])

    expected_revenue = (1.0 - pd_value) * ead * revenue_rate(cfg)
    el = pd_value * lgd * ead
    profit = expected_revenue - el
    grade = assign_grade(pd_value, cfg)

    if grade == hard_decline:
        decision = Decision.DECLINE
        reason = f"Risk grade {grade} is outside credit policy regardless of economics."
    elif pd_value <= threshold - band:
        decision = Decision.APPROVE
        reason = (
            f"PD {pd_value:.2%} is below the break-even PD of {threshold:.2%} "
            f"by more than the {band:.0%} review band."
        )
    elif pd_value >= threshold + band:
        decision = Decision.DECLINE
        reason = (
            f"PD {pd_value:.2%} exceeds the break-even PD of {threshold:.2%}; "
            f"expected loss outweighs expected margin."
        )
    else:
        decision = Decision.REVIEW
        reason = (
            f"PD {pd_value:.2%} is within the {band:.0%} review band around the "
            f"break-even PD of {threshold:.2%}. Manual underwriting required."
        )

    return DecisionResult(
        decision=decision,
        reason=reason,
        break_even_pd=threshold,
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
        "annual_net_margin": float(econ["annual_net_margin"]),
        "expected_life_years": float(econ["expected_life_years"]),
        "lifetime_revenue_rate": revenue_rate(cfg),
        "loss_aversion_multiplier": float(econ["loss_aversion_multiplier"]),
        "manual_review_margin": float(cfg["decision"]["manual_review_margin"]),
        "note": (
            "The approve/decline cut-off is derived from these economics, not "
            "chosen. It varies with LGD: better-collateralised exposures are "
            "tolerated at a higher PD."
        ),
    }
