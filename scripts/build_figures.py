"""Regenerate the figures used in README.md.

Run after retraining:

    python scripts/build_figures.py

The split is reproduced from config/model.yaml rather than hardcoded, so the
curves always describe the model that is actually committed. Figures are written
to docs/images/ and committed, so the README renders without anyone needing
matplotlib installed.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from loan_default.config import PROJECT_ROOT, get_settings, load_model_config
from loan_default.data.loader import load_dataset
from loan_default.models.calibration import reliability_table
from loan_default.models.registry import ModelRegistry
from loan_default.risk.grades import grade_summary

OUT = PROJECT_ROOT / "docs" / "images"

# A restrained palette. White background so the figures stay legible against
# both GitHub themes.
INK = "#1f2933"
MUTED = "#7b8794"
GOOD = "#2f6f4e"
BAD = "#a4303f"
ACCENT = "#2c5282"
GRID = "#e4e7eb"

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "600",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
    }
)


def _tidy(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)


def _splits():
    """Reproduce the exact train/calibration/test split used in training."""
    cfg = load_model_config()
    seed = int(cfg["seed"])
    val = cfg["validation"]
    dataset = load_dataset()
    X, y = dataset.X, dataset.y

    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=val["test_size"], stratify=y, random_state=seed
    )
    return X_test, y_test, seed


def figure_roc(model, metadata, X_test, y_test, seed):
    """The leaking signal against the honest model.

    These two curves are NOT on the same population, and the figure says so.
    The leaking indicator is only defined on the raw file, because removing the
    leakage means dropping the 24,123 rows whose missingness gave the answer
    away. Plotting them together shows the cost of the fix rather than hiding it.
    """
    honest = model.predict_proba(X_test[metadata.feature_columns])[:, 1]
    honest_auc = roc_auc_score(y_test, honest)

    raw = pd.read_csv(PROJECT_ROOT / "data" / "Loan_Default.csv")
    leak_indicator = raw["Interest_rate_spread"].isna().astype(int)
    y_raw = raw["Status"].astype(int)
    leak_auc = roc_auc_score(y_raw, leak_indicator)

    fig, ax = plt.subplots(figsize=(6.6, 5.4))

    fpr, tpr, _ = roc_curve(y_raw, leak_indicator)
    ax.plot(
        fpr,
        tpr,
        color=BAD,
        linewidth=2.4,
        label=f"Missingness indicator — AUC {leak_auc:.4f}\n(raw file, {len(raw):,} rows)",
    )

    fpr, tpr, _ = roc_curve(y_test, honest)
    ax.plot(
        fpr,
        tpr,
        color=GOOD,
        linewidth=2.4,
        label=f"Model, leakage removed — AUC {honest_auc:.4f}\n(held-out, {len(X_test):,} rows)",
    )

    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Random — AUC 0.5000")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One column's blankness was the answer")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    _tidy(ax)
    fig.savefig(OUT / "roc_leakage_vs_honest.png")
    plt.close(fig)
    return honest, honest_auc, leak_auc


def figure_calibration(model, metadata, X_test, y_test, honest, metrics):
    """Predicted PD against observed default rate, before and after isotonic.

    Both curves are shown because the interesting result is not that the
    calibrated model is close to the diagonal - it is that isotonic gives up a
    fraction of Brier score to buy a much smaller worst-case gap. Plotting only
    the calibrated curve would hide that trade.
    """
    comparison = metrics["calibration_comparison"]

    # cv="prefit" means the fitted base estimator lives on the calibrator.
    base = model.calibrated_classifiers_[0].estimator
    uncal = base.predict_proba(X_test[metadata.feature_columns])[:, 1]

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1.2, label="Perfect calibration")

    for scores, key, colour, marker in (
        (uncal, "uncalibrated", BAD, "s"),
        (np.asarray(honest), "isotonic", GOOD, "o"),
    ):
        table = reliability_table(np.asarray(y_test), np.asarray(scores), n_bins=10)
        stats = comparison[key]
        ax.plot(
            table["predicted"],
            table["observed"],
            marker=marker,
            linestyle="-",
            color=colour,
            linewidth=2,
            markersize=5,
            label=(
                f"{stats['label'].capitalize()} — worst gap "
                f"{stats['max_calibration_error'] * 100:.2f} pp"
            ),
        )

    ax.set_xlabel("Mean predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("A predicted 8% should default 8% of the time")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    upper = 0.85
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    _tidy(ax)
    fig.savefig(OUT / "calibration_curve.png")
    plt.close(fig)


def figure_grades(y_test, honest):
    """Observed default rate by grade, which must increase A through G."""
    summary = grade_summary(np.asarray(honest), np.asarray(y_test))
    summary = summary[summary["n"] > 0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.4))

    rates = summary["observed_default_rate"] * 100
    ax1.bar(summary["grade"], rates, color=ACCENT, width=0.62)
    ax1.set_ylabel("Observed default rate (%)")
    ax1.set_xlabel("Risk grade")
    ax1.set_title("Default rate rises monotonically A to G")
    for x, v in zip(summary["grade"], rates, strict=False):
        ax1.text(x, v + max(rates) * 0.02, f"{v:.1f}", ha="center", fontsize=8, color=INK)
    _tidy(ax1)

    ax2.bar(summary["grade"], summary["n"], color=MUTED, width=0.62)
    ax2.set_ylabel("Loans in grade")
    ax2.set_xlabel("Risk grade")
    ax2.set_title("Population by grade")
    _tidy(ax2)

    fig.tight_layout()
    fig.savefig(OUT / "risk_grades.png")
    plt.close(fig)
    return summary


def figure_pd_distribution(y_test, honest):
    """Separation without the piled-at-0-and-1 signature of a leaking model."""
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    y = np.asarray(y_test)
    bins = np.linspace(0, max(0.6, float(np.percentile(honest, 99.5))), 45)

    ax.hist(honest[y == 0], bins=bins, color=GOOD, alpha=0.65, label="Repaid", density=True)
    ax.hist(honest[y == 1], bins=bins, color=BAD, alpha=0.65, label="Defaulted", density=True)
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Density")
    ax.set_title("Overlapping distributions, as real credit data gives you")
    ax.legend(frameon=False, fontsize=9)
    _tidy(ax)
    fig.savefig(OUT / "pd_distribution.png")
    plt.close(fig)


def figure_importance(model, metadata, X_test):
    """What the honest model actually leans on."""
    from loan_default.models.explain import PredictionExplainer

    explainer = PredictionExplainer(model)
    if not explainer.available:
        return None

    importance = explainer.global_importance(X_test[metadata.feature_columns], sample=2000)
    top = importance.head(12).iloc[::-1]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.barh(top["feature"], top["mean_abs_shap"], color=ACCENT, height=0.68)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Signal is spread across many features, not one")
    ax.grid(axis="y", visible=False)
    _tidy(ax)
    fig.savefig(OUT / "feature_importance.png")
    plt.close(fig)
    return importance


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(get_settings().artifacts_dir)
    model, metadata, metrics = registry.load("latest")

    X_test, y_test, seed = _splits()
    print(f"test set: {len(X_test):,} loans, {y_test.mean():.2%} default rate")

    honest, honest_auc, leak_auc = figure_roc(model, metadata, X_test, y_test, seed)
    print(f"  roc_leakage_vs_honest.png   leaking AUC {leak_auc:.4f} vs honest {honest_auc:.4f}")

    figure_calibration(model, metadata, X_test, y_test, honest, metrics)
    print("  calibration_curve.png")

    summary = figure_grades(y_test, honest)
    print(f"  risk_grades.png             {len(summary)} populated grades")

    figure_pd_distribution(y_test, honest)
    print("  pd_distribution.png")

    if figure_importance(model, metadata, X_test) is not None:
        print("  feature_importance.png")

    (OUT / "figures.json").write_text(
        json.dumps(
            {
                "model_version": metadata.model_version,
                "test_rows": int(len(X_test)),
                "honest_roc_auc": round(float(honest_auc), 4),
                "leaking_roc_auc": round(float(leak_auc), 4),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwritten to {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
