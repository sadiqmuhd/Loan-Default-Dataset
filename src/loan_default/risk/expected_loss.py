"""Expected Loss: EL = PD x LGD x EAD.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS MODULE.

Only PD is modelled. LGD and EAD are *derived under stated assumptions*, because
the dataset contains no recovery cash flows, no workout costs, no
time-to-resolution and no balance history. Every assumption is in
config/risk_policy.yaml and is echoed back on every API response, so a consumer
of the number can always see what it rests on.

Presenting the LGD proxy as an estimated LGD model would be fabrication. It is
labelled a proxy everywhere it appears.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from loan_default.config import load_risk_policy


@dataclass
class LossComponents:
    """The decomposed expected loss for one exposure."""

    pd: float
    lgd: float
    ead: float
    expected_loss: float
    expected_loss_rate: float  # EL / EAD
    collateral_value: float | None
    lgd_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------- EAD


def exposure_at_default(loan_amount, policy: dict | None = None):
    """EAD for a fully-drawn term loan scored at origination.

    EAD equals the origination amount. Credit Conversion Factor modelling is not
    applicable: there is no undrawn commitment and no balance history in this
    dataset. This is a correct treatment for at-origination decisioning, not a
    compromise.
    """
    cfg = (policy or load_risk_policy())["ead"]
    return np.asarray(loan_amount, dtype=float) * float(cfg["utilisation"])


# --------------------------------------------------------------------- LGD


def loss_given_default(
    exposure,
    property_value=None,
    policy: dict | None = None,
) -> np.ndarray:
    """Collateral-based LGD PROXY. Not an estimated LGD model.

        LGD = clip(1 - (property_value * (1 - haircut) * (1 - cost)) / EAD,
                   floor, ceiling)

    Rationale for a collateral approach: 99.98% of this book is secured on
    residential property (``Secured_by == "home"``), so recovery is dominated by
    collateral value. Where ``property_value`` is unavailable, a flat fallback
    LGD is applied.
    """
    cfg = (policy or load_risk_policy())["lgd"]
    ead = np.asarray(exposure, dtype=float)

    if property_value is None:
        return np.full(ead.shape, float(cfg["fallback_lgd"]))

    collateral = np.asarray(property_value, dtype=float)
    recoverable = (
        collateral
        * (1.0 - float(cfg["distressed_sale_haircut"]))
        * (1.0 - float(cfg["workout_cost_rate"]))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        lgd = 1.0 - np.divide(recoverable, ead, out=np.full_like(ead, np.nan), where=ead > 0)

    lgd = np.where(np.isnan(lgd), float(cfg["fallback_lgd"]), lgd)
    return np.clip(lgd, float(cfg["floor"]), float(cfg["ceiling"]))


# ---------------------------------------------------------------------- EL


def expected_loss(
    pd_values,
    loan_amount,
    property_value=None,
    policy: dict | None = None,
) -> dict[str, np.ndarray]:
    """Vectorised EL = PD x LGD x EAD, returning every component."""
    cfg = policy or load_risk_policy()
    pd_arr = np.asarray(pd_values, dtype=float)
    ead = exposure_at_default(loan_amount, cfg)
    lgd = loss_given_default(ead, property_value, cfg)
    el = pd_arr * lgd * ead
    with np.errstate(divide="ignore", invalid="ignore"):
        el_rate = np.divide(el, ead, out=np.zeros_like(el), where=ead > 0)
    return {
        "pd": pd_arr,
        "lgd": lgd,
        "ead": ead,
        "expected_loss": el,
        "expected_loss_rate": el_rate,
    }


def compute_loss_components(
    pd_value: float,
    loan_amount: float,
    property_value: float | None = None,
    policy: dict | None = None,
) -> LossComponents:
    """Single-exposure convenience wrapper, used by the API."""
    cfg = policy or load_risk_policy()
    collateral: float | None = None
    if property_value is not None and np.isfinite(property_value):
        collateral = float(property_value)
    result = expected_loss(
        [pd_value],
        [loan_amount],
        None if collateral is None else [collateral],
        cfg,
    )
    return LossComponents(
        pd=float(result["pd"][0]),
        lgd=float(result["lgd"][0]),
        ead=float(result["ead"][0]),
        expected_loss=float(result["expected_loss"][0]),
        expected_loss_rate=float(result["expected_loss_rate"][0]),
        collateral_value=collateral,
        lgd_method="collateral_proxy" if collateral is not None else "fallback_flat_rate",
    )


def assumption_disclosure(policy: dict | None = None) -> dict[str, Any]:
    """The assumption set echoed on every response that quotes an EL figure."""
    cfg = policy or load_risk_policy()
    lgd_cfg, ead_cfg = cfg["lgd"], cfg["ead"]
    return {
        "lgd_is_modelled": False,
        "lgd_method": "collateral_proxy",
        "lgd_note": (
            "LGD is a collateral-based proxy, not an estimated model. The dataset "
            "contains no recovery cash flows, workout costs or resolution times."
        ),
        "distressed_sale_haircut": float(lgd_cfg["distressed_sale_haircut"]),
        "workout_cost_rate": float(lgd_cfg["workout_cost_rate"]),
        "lgd_floor": float(lgd_cfg["floor"]),
        "fallback_lgd": float(lgd_cfg["fallback_lgd"]),
        "ead_method": str(ead_cfg["method"]),
        "ead_note": (
            "EAD equals the origination amount. CCF modelling is not applicable to "
            "fully-drawn term loans."
        ),
    }
