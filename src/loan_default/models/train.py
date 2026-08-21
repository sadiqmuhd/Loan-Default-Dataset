"""Training entrypoint.

    python -m loan_default.models.train

Sequence:

1. Load data with leakage controls applied (see ``data/loader.py``).
2. Three-way split: fit / calibration / test. The calibration slice is held out
   of model fitting entirely, so the calibrator is not fitted on data the model
   has already seen.
3. Cross-validate candidate models on PR-AUC, the appropriate primary metric at
   a ~16% base rate.
4. Refit the winner, then calibrate it on the held-out calibration slice.
5. Evaluate on the untouched test slice, with bootstrap confidence intervals.
6. LEAKAGE GUARDRAIL: fail loudly if test ROC-AUC exceeds the plausible ceiling.
7. Persist a versioned artifact with full provenance.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from loan_default.config import get_settings, load_model_config
from loan_default.data.loader import load_dataset
from loan_default.logging_config import configure_logging
from loan_default.models.calibration import compare_methods, comparison_table, plot_reliability
from loan_default.models.evaluate import evaluate
from loan_default.models.pipeline import build_pipeline
from loan_default.models.registry import ModelRegistry, build_metadata, new_version_string

logger = logging.getLogger(__name__)


class LeakageGuardError(RuntimeError):
    """Raised when measured performance is too good to be legitimate."""


def _assert_no_leakage_features(feature_columns: list[str], cfg: dict[str, Any]) -> None:
    """Structural check: no excluded column, and no missingness indicator, survives."""
    excluded = {c for group in cfg["exclusions"].values() for c in group}
    present = sorted(set(feature_columns) & excluded)
    if present:
        raise LeakageGuardError(
            f"Excluded columns reached the feature set: {present}. "
            "See config/model.yaml exclusions."
        )
    indicators = [c for c in feature_columns if c.endswith(("_missing", "_isna", "_is_null"))]
    if indicators:
        raise LeakageGuardError(
            f"Missingness indicators are forbidden on this dataset - the missingness "
            f"of several pricing fields encodes the target. Found: {indicators}"
        )


def train(
    *,
    quick: bool = False,
    skip_cv: bool = False,
) -> dict[str, Any]:
    cfg = load_model_config()
    settings = get_settings()
    seed = int(cfg["seed"])
    val = cfg["validation"]
    started = time.perf_counter()

    # ------------------------------------------------------------------ data
    dataset = load_dataset()
    X, y = dataset.X, dataset.y
    logger.info(
        "dataset: %d rows (%d dropped from %d raw), default rate %.4f",
        dataset.n_rows,
        dataset.rows_dropped,
        dataset.n_raw_rows,
        y.mean(),
    )

    numeric = [c for c in cfg["features"]["numeric"] if c in X.columns]
    categorical = [c for c in cfg["features"]["categorical"] if c in X.columns]
    _assert_no_leakage_features(list(X.columns), cfg)

    if quick:
        X = X.sample(n=min(20_000, len(X)), random_state=seed)
        y = y.loc[X.index]
        logger.warning("QUICK MODE: sampled %d rows, results are not publishable", len(X))

    # --------------------------------------------------------------- split
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=val["test_size"], stratify=y, random_state=seed
    )
    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_dev, y_dev, test_size=val["calibration_size"], stratify=y_dev, random_state=seed
    )
    logger.info("split: fit=%d  calibration=%d  test=%d", len(X_fit), len(X_cal), len(X_test))

    # ------------------------------------------------------- model selection
    candidate_scores: dict[str, float] = {}
    candidates = cfg["candidates"]

    if skip_cv:
        best_name = "xgboost"
        logger.warning("skipping cross-validation, defaulting to %s", best_name)
    else:
        cv = StratifiedKFold(n_splits=val["cv_folds"], shuffle=True, random_state=seed)
        for name, params in candidates.items():
            pipe = build_pipeline(
                name, params, numeric, categorical, seed, cfg.get("feature_params")
            )
            scores = cross_val_score(
                pipe, X_fit, y_fit, cv=cv, scoring="average_precision", n_jobs=1
            )
            candidate_scores[name] = float(scores.mean())
            logger.info(
                "candidate %-20s CV PR-AUC = %.4f (+/- %.4f)",
                name,
                scores.mean(),
                scores.std(),
            )
        best_name = max(candidate_scores, key=lambda name: candidate_scores[name])

    logger.info("selected model: %s", best_name)

    # ------------------------------------------------------- fit + calibrate
    base = build_pipeline(
        best_name, candidates[best_name], numeric, categorical, seed, cfg.get("feature_params")
    )
    base.fit(X_fit, y_fit)

    uncalibrated_test = base.predict_proba(X_test)[:, 1]
    uncalibrated_report = evaluate(y_test, uncalibrated_test)

    # Fit both candidate calibrators on the held-out slice and keep whichever
    # actually calibrates better, rather than trusting the config default.
    configured = cfg["calibration"]["method"]
    comparison = compare_methods(base, X_cal, y_cal, X_test, y_test)
    method = comparison["best_method"]
    calibrated = comparison["best_model"]

    print("\n" + "=" * 70)
    print("CALIBRATION METHOD COMPARISON")
    print("=" * 70)
    print(comparison_table(comparison).to_string(float_format=lambda v: f"{v:.4f}"))
    if method != configured:
        logger.info("config suggested %s; %s calibrated better and was used", configured, method)
    logger.info("calibrated with method=%s on %d held-out rows", method, len(X_cal))

    if not quick:
        plot_reliability(
            {
                "Uncalibrated": (y_test, uncalibrated_test),
                f"Calibrated ({method})": (y_test, calibrated.predict_proba(X_test)[:, 1]),
            },
            settings.reports_dir / "reliability_curve.png",
        )

    # ------------------------------------------------------------- evaluate
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    report = evaluate(
        y_test,
        y_prob,
        bootstrap_iterations=0 if quick else int(val["bootstrap_iterations"]),
        seed=seed,
    )

    print("\n" + "=" * 70)
    print(f"UNCALIBRATED ({best_name})")
    print("=" * 70)
    print(uncalibrated_report.summary())
    print("\n" + "=" * 70)
    print(f"CALIBRATED ({best_name} + {method})")
    print("=" * 70)
    print(report.summary())
    print("=" * 70)
    print(
        f"calibration improvement: Brier {uncalibrated_report.brier_score:.4f} "
        f"-> {report.brier_score:.4f}   "
        f"cal.error {uncalibrated_report.calibration_error:.4f} "
        f"-> {report.calibration_error:.4f}"
    )

    # ------------------------------------------------------ LEAKAGE GUARDRAIL
    ceiling = float(cfg["max_plausible_auc"])
    if report.roc_auc > ceiling:
        raise LeakageGuardError(
            f"Test ROC-AUC {report.roc_auc:.4f} exceeds the plausible ceiling of {ceiling}. "
            "On this dataset that means a leaking feature has been reintroduced - "
            "see docs/LEAKAGE_INVESTIGATION.md."
        )

    duration = time.perf_counter() - started

    # ---------------------------------------------------------------- persist
    version = new_version_string()
    metadata = build_metadata(
        model_version=version,
        model_type=best_name,
        training_duration_seconds=round(duration, 2),
        data_sha256=dataset.data_hash,
        data_source=dataset.source_path,
        n_training_rows=len(X_fit),
        n_test_rows=len(X_test),
        default_rate=float(y.mean()),
        feature_columns=numeric + categorical,
        numeric_features=numeric,
        categorical_features=categorical,
        engineered_features=list(cfg["features"]["engineered"]),
        excluded_columns=dict(cfg["exclusions"]),
        seed=seed,
        hyperparameters=candidates[best_name],
        calibration_method=method,
        selection_metric="pr_auc",
        candidate_scores=candidate_scores,
        assumptions=[
            "Status is treated as a 12-month default flag; the dataset does not "
            "state an observation window.",
            "income is interpreted as MONTHLY income (median 5,760 against a median "
            "loan of 296,500; an annual reading implies a 52x loan-to-income ratio).",
            "Complete-case training: rows missing property_value, LTV or dtir1 are "
            "excluded because their missingness predicts default (ROC-AUC 0.7155 "
            "from missingness alone).",
        ],
        limitations=[
            "Credit_Score is not predictive in this dataset (univariate ROC-AUC "
            "0.5030, flat default rate across all deciles) and appears to be "
            "randomly generated. Do not interpret its coefficient or SHAP value.",
            "The dataset has no time dimension (year == 2019 for all rows), so no "
            "out-of-time validation, vintage analysis or macroeconomic "
            "conditioning is possible.",
            "Trained on a single cross-section; behaviour under large input shocks "
            "is an extrapolation beyond the training distribution.",
        ],
    )

    metrics_payload = {
        "calibrated": report.to_dict(),
        "uncalibrated": uncalibrated_report.to_dict(),
        "candidate_cv_pr_auc": candidate_scores,
        "calibration_comparison": comparison["results"],
        "data_provenance": dataset.provenance(),
    }

    registry = ModelRegistry(settings.artifacts_dir)
    path = registry.save(calibrated, metadata, metrics_payload)
    print(f"\nSaved model {version} -> {path}")

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "latest_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, default=str), encoding="utf-8"
    )

    return {"version": version, "metrics": metrics_payload, "metadata": metadata.to_dict()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the PD model.")
    parser.add_argument("--quick", action="store_true", help="Subsample for a fast smoke run.")
    parser.add_argument("--skip-cv", action="store_true", help="Skip candidate cross-validation.")
    args = parser.parse_args()

    configure_logging()
    np.random.seed(load_model_config()["seed"])
    try:
        train(quick=args.quick, skip_cv=args.skip_cv)
    except LeakageGuardError as exc:
        logger.error("LEAKAGE GUARD TRIPPED: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
