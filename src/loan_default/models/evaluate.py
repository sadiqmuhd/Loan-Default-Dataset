"""Model evaluation for a PD model.

Discrimination (can the model rank borrowers?) and calibration (are the numbers
actually probabilities?) are reported separately, because a PD is consumed as a
number in an expected-loss calculation, not just as a ranking.

Accuracy is deliberately not among them. At a 16% default rate, predicting "no
default" for every applicant scores 0.84 while being useless.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    roc_auc_score,
    roc_curve,
)


@dataclass
class EvaluationReport:
    """Full metric suite for a binary PD model."""

    n: int
    base_rate: float
    # Discrimination
    roc_auc: float
    gini: float
    pr_auc: float
    ks_statistic: float
    # Calibration / probabilistic accuracy
    brier_score: float
    log_loss: float
    mean_predicted_pd: float
    calibration_error: float  # mean |observed - predicted| across bins
    max_calibration_error: float
    # Uncertainty
    roc_auc_ci: tuple[float, float] | None = None
    pr_auc_ci: tuple[float, float] | None = None
    # Detail
    calibration_bins: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        ci = ""
        if self.roc_auc_ci:
            ci = f"  95% CI [{self.roc_auc_ci[0]:.4f}, {self.roc_auc_ci[1]:.4f}]"
        return (
            f"  n = {self.n:,}   base rate = {self.base_rate:.4f}\n"
            f"  ROC-AUC      {self.roc_auc:.4f}{ci}\n"
            f"  Gini         {self.gini:.4f}\n"
            f"  PR-AUC       {self.pr_auc:.4f}   (baseline {self.base_rate:.4f})\n"
            f"  KS           {self.ks_statistic:.4f}\n"
            f"  Brier        {self.brier_score:.4f}\n"
            f"  Log loss     {self.log_loss:.4f}\n"
            f"  Mean PD      {self.mean_predicted_pd:.4f}   vs observed {self.base_rate:.4f}\n"
            f"  Cal. error   {self.calibration_error:.4f}  (max {self.max_calibration_error:.4f})"
        )


def ks_statistic(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation between the good and bad score distributions.

    Standard in retail credit scorecards; reported alongside Gini in most banks.
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Observed vs predicted default rate by predicted-PD decile.

    Quantile bins rather than equal-width, so every bin carries mass.
    """
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    try:
        df["bin"] = pd.qcut(df["p"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        df["bin"] = 0
    grouped = df.groupby("bin", observed=True).agg(
        n=("y", "size"),
        predicted=("p", "mean"),
        observed=("y", "mean"),
        pd_min=("p", "min"),
        pd_max=("p", "max"),
    )
    grouped["gap"] = grouped["observed"] - grouped["predicted"]
    return grouped.reset_index()


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_iterations: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a ranking metric."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = []
    for _ in range(n_iterations):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        if yt.min() == yt.max():  # degenerate resample, no positives or no negatives
            continue
        stats.append(metric_fn(yt, y_prob[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def evaluate(
    y_true,
    y_prob,
    *,
    n_bins: int = 10,
    bootstrap_iterations: int = 0,
    seed: int = 42,
) -> EvaluationReport:
    """Compute the full PD metric suite.

    Set ``bootstrap_iterations`` > 0 to attach 95% confidence intervals to the
    ranking metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if y_true.min() == y_true.max():
        raise ValueError("Evaluation requires both classes to be present.")

    auc = float(roc_auc_score(y_true, y_prob))
    pr_auc = float(average_precision_score(y_true, y_prob))
    cal = calibration_table(y_true, y_prob, n_bins)
    gaps = cal["gap"].abs()

    roc_ci = pr_ci = None
    if bootstrap_iterations > 0:
        roc_ci = _bootstrap_ci(y_true, y_prob, roc_auc_score, bootstrap_iterations, seed)
        pr_ci = _bootstrap_ci(
            y_true, y_prob, average_precision_score, bootstrap_iterations, seed + 1
        )

    return EvaluationReport(
        n=int(len(y_true)),
        base_rate=float(y_true.mean()),
        roc_auc=auc,
        gini=float(2 * auc - 1),
        pr_auc=pr_auc,
        ks_statistic=ks_statistic(y_true, y_prob),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        log_loss=float(log_loss(y_true, np.clip(y_prob, 1e-15, 1 - 1e-15))),
        mean_predicted_pd=float(y_prob.mean()),
        calibration_error=float(gaps.mean()),
        max_calibration_error=float(gaps.max()),
        roc_auc_ci=roc_ci,
        pr_auc_ci=pr_ci,
        calibration_bins=cast(list[dict[str, float]], cal.to_dict(orient="records")),
    )


def confusion_at_threshold(y_true, y_prob, threshold: float) -> dict[str, Any]:
    """Confusion matrix and rates at a specific decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) else 0.0,
        "approval_rate": float((y_pred == 0).mean()),
        "bad_rate_in_approved": float(y_true[y_pred == 0].mean()) if (y_pred == 0).any() else 0.0,
    }
