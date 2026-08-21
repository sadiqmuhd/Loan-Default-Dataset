"""Probability calibration and the diagnostics that justify it.

Discrimination and calibration are different properties and a model can have one
without the other. Ranking borrowers correctly is enough if you only ever sort
them; it is not enough here, because the PD is multiplied by LGD and EAD to get
a currency figure. A model that ranks perfectly but says 5% when the true rate
is 15% will understate expected loss by a factor of three.

So the choice of calibrator is made by measurement rather than by preference:
``compare_methods`` fits both candidates on the same held-out slice and reports
what each does to Brier score and to calibration error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)

Method = Literal["isotonic", "sigmoid"]

# "sigmoid" is scikit-learn's name for Platt scaling.
METHOD_LABELS: dict[str, str] = {
    "isotonic": "Isotonic regression",
    "sigmoid": "Platt scaling (sigmoid)",
}


@dataclass
class CalibrationResult:
    """What one calibration method did to the held-out predictions."""

    method: str
    label: str
    brier_score: float
    calibration_error: float
    max_calibration_error: float
    roc_auc: float
    mean_predicted_pd: float
    observed_default_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "label": self.label,
            "brier_score": self.brier_score,
            "calibration_error": self.calibration_error,
            "max_calibration_error": self.max_calibration_error,
            "roc_auc": self.roc_auc,
            "mean_predicted_pd": self.mean_predicted_pd,
            "observed_default_rate": self.observed_default_rate,
        }


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Predicted versus observed default rate, in quantile bins.

    Quantile rather than equal-width bins, so each row carries comparable mass;
    with a right-skewed PD distribution equal-width bins leave the top ones
    nearly empty and the curve becomes noise.
    """
    frame = pd.DataFrame(
        {"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, dtype=float)}
    )
    try:
        frame["bin"] = pd.qcut(frame["p"], n_bins, labels=False, duplicates="drop")
    except ValueError:
        frame["bin"] = 0

    table = frame.groupby("bin", observed=True).agg(
        n=("y", "size"),
        predicted=("p", "mean"),
        observed=("y", "mean"),
        lower=("p", "min"),
        upper=("p", "max"),
    )
    table["gap"] = table["observed"] - table["predicted"]
    return table.reset_index(drop=True)


def calibration_error(y_true, y_prob, n_bins: int = 10) -> tuple[float, float]:
    """Mean and maximum absolute gap between predicted and observed rates."""
    gaps = reliability_table(y_true, y_prob, n_bins)["gap"].abs()
    return float(gaps.mean()), float(gaps.max())


def summarise(y_true, y_prob, method: str, n_bins: int = 10) -> CalibrationResult:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    mean_gap, max_gap = calibration_error(y_true, y_prob, n_bins)
    return CalibrationResult(
        method=method,
        label=METHOD_LABELS.get(method, method),
        brier_score=float(brier_score_loss(y_true, y_prob)),
        calibration_error=mean_gap,
        max_calibration_error=max_gap,
        roc_auc=float(roc_auc_score(y_true, y_prob)),
        mean_predicted_pd=float(y_prob.mean()),
        observed_default_rate=float(y_true.mean()),
    )


def calibrate(base_model, X_cal, y_cal, method: Method = "isotonic"):
    """Wrap an already-fitted model in a calibrator fitted on held-out data.

    ``FrozenEstimator`` stops scikit-learn refitting the base model, so the
    calibrator only ever sees predictions from a model that did not train on
    ``X_cal``. Calibrating on training data produces a calibrator fitted to
    already-overconfident scores and quietly does nothing useful.
    """
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def compare_methods(
    base_model,
    X_cal,
    y_cal,
    X_test,
    y_test,
    methods: tuple[Method, ...] = ("isotonic", "sigmoid"),
    n_bins: int = 10,
) -> dict[str, Any]:
    """Fit every candidate calibrator and measure them on the same test set.

    Selection is on calibration error rather than Brier score. Brier mixes
    discrimination and calibration together, so it barely moves when only the
    latter improves, and it would hide the difference we are trying to measure.
    """
    uncalibrated = base_model.predict_proba(X_test)[:, 1]
    results = {"uncalibrated": summarise(y_test, uncalibrated, "uncalibrated", n_bins)}
    fitted: dict[str, Any] = {}

    for method in methods:
        model = calibrate(base_model, X_cal, y_cal, method)
        probabilities = model.predict_proba(X_test)[:, 1]
        results[method] = summarise(y_test, probabilities, method, n_bins)
        fitted[method] = model
        logger.info(
            "%s: Brier %.4f, calibration error %.4f (max %.4f)",
            METHOD_LABELS.get(method, method),
            results[method].brier_score,
            results[method].calibration_error,
            results[method].max_calibration_error,
        )

    candidates = {m: r for m, r in results.items() if m in methods}
    best = min(candidates, key=lambda m: candidates[m].calibration_error)
    logger.info("selected %s", METHOD_LABELS.get(best, best))

    return {
        "best_method": best,
        "best_model": fitted[best],
        "results": {name: result.to_dict() for name, result in results.items()},
        "fitted": fitted,
        "selection_metric": "calibration_error",
    }


def comparison_table(comparison: dict[str, Any]) -> pd.DataFrame:
    """The method comparison as a readable frame, for reports and notebooks."""
    rows = [
        {"method": name, **{k: v for k, v in result.items() if k != "method"}}
        for name, result in comparison["results"].items()
    ]
    return pd.DataFrame(rows).set_index("method")


def plot_reliability(
    curves: dict[str, tuple],
    output_path: Path,
    title: str = "Reliability: predicted vs observed default rate",
    n_bins: int = 10,
) -> Path | None:
    """Write a reliability diagram. ``curves`` maps a label to ``(y_true, y_prob)``.

    Returns ``None`` when matplotlib is unavailable, since plotting is a
    reporting convenience and must not be a hard runtime dependency of the API.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping reliability plot")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax, hist_ax) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    for label, (y_true, y_prob) in curves.items():
        table = reliability_table(y_true, y_prob, n_bins)
        ax.plot(table["predicted"], table["observed"], marker="o", linewidth=1.5, label=label)
        hist_ax.hist(np.asarray(y_prob, dtype=float), bins=40, alpha=0.5, label=label)

    limit = max(
        0.05,
        min(1.0, max(float(np.max(p)) for _, p in curves.values()) * 1.05),
    )
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_ylabel("Observed default rate")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    hist_ax.set_xlabel("Predicted probability of default")
    hist_ax.set_ylabel("Count")
    hist_ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    logger.info("wrote reliability plot to %s", output_path)
    return output_path
