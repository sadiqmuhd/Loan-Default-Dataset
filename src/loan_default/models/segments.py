"""Performance broken down by portfolio segment.

A single headline AUC hides a model that ranks well overall while being close to
useless in one region or product. Since lending decisions are made per applicant
and not per portfolio, a segment where the model does not discriminate is a
segment where it should not be used unsupervised.

Two things are reported per segment, because they fail independently:

* discrimination (AUC) - can the model rank borrowers within this segment?
* calibration (predicted vs observed) - are its PDs right in level here?

A segment can rank perfectly and still be badly miscalibrated, which matters
because the PD feeds an expected-loss figure in currency.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)

# Below this, a segment AUC is too noisy to act on.
MIN_SEGMENT_SIZE = 200


def segment_performance(
    frame: pd.DataFrame,
    y_true,
    y_prob,
    column: str,
    min_size: int = MIN_SEGMENT_SIZE,
) -> pd.DataFrame:
    """Discrimination and calibration within each level of ``column``."""
    data = pd.DataFrame(
        {
            "segment": np.asarray(frame[column]),
            "y": np.asarray(y_true).astype(int),
            "p": np.asarray(y_prob, dtype=float),
        }
    )

    rows: list[dict[str, Any]] = []
    for level, group in data.groupby("segment", observed=True):
        n = len(group)
        positives = int(group["y"].sum())
        # AUC is undefined without both classes present.
        evaluable = n >= min_size and 0 < positives < n

        rows.append(
            {
                "segment": level,
                "n": n,
                "share": n / len(data),
                "default_rate": float(group["y"].mean()),
                "mean_pd": float(group["p"].mean()),
                "calibration_gap": float(group["y"].mean() - group["p"].mean()),
                "roc_auc": float(roc_auc_score(group["y"], group["p"])) if evaluable else np.nan,
                "pr_auc": (
                    float(average_precision_score(group["y"], group["p"])) if evaluable else np.nan
                ),
                "brier": float(brier_score_loss(group["y"], group["p"])) if evaluable else np.nan,
                "evaluable": evaluable,
            }
        )

    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def segment_report(
    frame: pd.DataFrame,
    y_true,
    y_prob,
    columns: tuple[str, ...] = ("Region", "loan_purpose", "occupancy_type", "credit_type"),
    min_size: int = MIN_SEGMENT_SIZE,
    auc_floor: float = 0.65,
    calibration_tolerance: float = 0.03,
) -> dict[str, Any]:
    """Segment performance across several dimensions, with flags.

    ``auc_floor`` and ``calibration_tolerance`` are review triggers rather than
    pass/fail thresholds: a flagged segment is one a model risk reviewer should
    look at, not necessarily one that is broken.
    """
    overall_auc = float(roc_auc_score(np.asarray(y_true).astype(int), y_prob))
    report: dict[str, Any] = {
        "overall_roc_auc": overall_auc,
        "min_segment_size": min_size,
        "auc_floor": auc_floor,
        "calibration_tolerance": calibration_tolerance,
        "dimensions": {},
    }
    flagged: list[dict[str, Any]] = []

    for column in columns:
        if column not in frame.columns:
            continue
        table = segment_performance(frame, y_true, y_prob, column, min_size)

        records: list[dict[str, Any]] = table.to_dict(orient="records")  # type: ignore[assignment]
        for row in records:
            if not row["evaluable"]:
                continue
            auc = float(row["roc_auc"])
            gap = float(row["calibration_gap"])
            reasons: list[str] = []
            if auc < auc_floor:
                reasons.append(f"AUC {auc:.3f} below floor {auc_floor}")
            if abs(gap) > calibration_tolerance:
                reasons.append(f"observed default rate is {gap:+.3f} from predicted")
            if reasons:
                flagged.append(
                    {
                        "dimension": column,
                        "segment": str(row["segment"]),
                        "n": int(row["n"]),
                        "roc_auc": auc,
                        "calibration_gap": gap,
                        "reasons": reasons,
                    }
                )

        report["dimensions"][column] = records

    report["flagged"] = flagged
    report["n_flagged"] = len(flagged)
    logger.info(
        "segment review: %d segment(s) flagged across %d dimension(s)",
        len(flagged),
        len(report["dimensions"]),
    )
    return report


def format_report(report: dict[str, Any]) -> str:
    """Readable rendering for the console and for reports/."""
    lines = [f"Overall ROC-AUC: {report['overall_roc_auc']:.4f}", ""]
    for column, rows in report["dimensions"].items():
        lines.append(f"--- {column} ---")
        table = pd.DataFrame(rows)
        display = table[["segment", "n", "default_rate", "mean_pd", "calibration_gap", "roc_auc"]]
        lines.append(display.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        lines.append("")

    if report["flagged"]:
        lines.append(f"FLAGGED FOR REVIEW ({report['n_flagged']}):")
        for item in report["flagged"]:
            lines.append(
                f"  {item['dimension']}={item['segment']} (n={item['n']:,}): "
                + "; ".join(item["reasons"])
            )
    else:
        lines.append("No segment breached the review thresholds.")
    return "\n".join(lines)
