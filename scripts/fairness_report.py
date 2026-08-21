"""Fair lending analysis.

Excluding protected characteristics from the model is necessary but not
sufficient. A model can still produce disparate outcomes through correlated
proxies, so outcomes must be MEASURED across protected groups even though the
model never sees them.

That is exactly what this script does: it scores the portfolio with a model
trained without ``Gender`` or ``age``, then joins the protected attributes back
on purely for measurement, and reports:

  * approval rate by group
  * mean PD by group
  * the adverse impact ratio (the "four-fifths rule")
  * the measured cost in ROC-AUC of excluding the attributes at all

Usage:
    python scripts/fairness_report.py
Writes:
    reports/fairness_report.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from credit_risk.config import get_settings, load_model_config, load_risk_policy  # noqa: E402
from credit_risk.data.loader import load_dataset  # noqa: E402
from credit_risk.logging_config import configure_logging  # noqa: E402
from credit_risk.models.registry import ModelRegistry  # noqa: E402
from credit_risk.risk.grades import assign_grades  # noqa: E402
from credit_risk.risk.policy import break_even_pd  # noqa: E402

logger = logging.getLogger(__name__)

# The four-fifths rule: a selection rate below 80% of the most-favoured group's
# rate is the conventional threshold for evidence of adverse impact
# (29 CFR 1607.4(D), applied here by analogy to credit decisioning).
ADVERSE_IMPACT_THRESHOLD = 0.80


def group_outcomes(df: pd.DataFrame, attribute: str) -> list[dict]:
    """Approval rate, mean PD and adverse impact ratio by group."""
    grouped = df.groupby(attribute, observed=True).agg(
        n=("pd", "size"),
        mean_pd=("pd", "mean"),
        approval_rate=("approved", "mean"),
        observed_default_rate=("y", "mean"),
    )
    best = grouped["approval_rate"].max()
    grouped["adverse_impact_ratio"] = grouped["approval_rate"] / best if best > 0 else np.nan
    grouped["flag"] = grouped["adverse_impact_ratio"] < ADVERSE_IMPACT_THRESHOLD
    return grouped.reset_index().to_dict(orient="records")


def cost_of_exclusion(seed: int = 42) -> dict:
    """Retrain WITH the protected attributes to measure what exclusion costs.

    Being able to quote this number is the point: it turns a compliance
    constraint into a quantified, defensible trade-off.
    """
    from sklearn.model_selection import train_test_split

    from credit_risk.models.pipeline import build_pipeline

    cfg = load_model_config()
    raw = pd.read_csv(get_settings().data_path)
    raw = raw.dropna(subset=cfg["complete_case_columns"] + [cfg["target"]])

    leakage = cfg["exclusions"]["leakage"] + cfg["exclusions"]["identifiers"]
    y = raw[cfg["target"]].astype(int)

    results = {}
    for label, extra_drop in (("without_protected", ["Gender", "age"]), ("with_protected", [])):
        X = raw.drop(columns=[cfg["target"], *leakage, *extra_drop], errors="ignore")
        numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        categorical = [c for c in X.columns if c not in numeric]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=seed
        )
        pipeline = build_pipeline(
            "xgboost",
            cfg["candidates"]["xgboost"],
            numeric,
            categorical,
            seed,
            cfg.get("feature_params"),
        )
        pipeline.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
        results[label] = float(auc)
        logger.info("%s: ROC-AUC %.4f", label, auc)

    delta = results["with_protected"] - results["without_protected"]
    results["auc_cost_of_exclusion"] = float(delta)
    results["auc_cost_basis_points"] = float(delta * 10_000)
    return results


def main() -> int:
    configure_logging()
    settings = get_settings()
    policy = load_risk_policy()

    if not settings.data_path.exists():
        logger.error("dataset not found at %s", settings.data_path)
        return 1

    model, metadata, _ = ModelRegistry(settings.artifacts_dir).load("latest")

    # Score with the compliant model, then attach protected attributes for
    # measurement only. They are never inputs.
    dataset = load_dataset()
    raw = pd.read_csv(settings.data_path)
    raw = raw[raw[load_model_config()["target"]].notna()]
    raw = raw.dropna(subset=load_model_config()["complete_case_columns"]).reset_index(drop=True)

    frame = dataset.X[metadata.feature_columns]
    pd_values = model.predict_proba(frame)[:, 1]

    threshold = break_even_pd(policy["lgd"]["fallback_lgd"], policy)
    scored = pd.DataFrame(
        {
            "pd": pd_values,
            "y": dataset.y.to_numpy(),
            "grade": assign_grades(pd_values, policy),
            "approved": (pd_values <= threshold).astype(int),
            "Gender": raw["Gender"].to_numpy(),
            "age": raw["age"].to_numpy(),
            "Region": raw["Region"].to_numpy(),
        }
    )

    report = {
        "model_version": metadata.model_version,
        "n_scored": int(len(scored)),
        "decision_threshold_pd": float(threshold),
        "overall_approval_rate": float(scored["approved"].mean()),
        "protected_attributes_in_model": [],
        "adverse_impact_threshold": ADVERSE_IMPACT_THRESHOLD,
        "groups": {
            attribute: group_outcomes(scored, attribute)
            for attribute in ("Gender", "age", "Region")
        },
        "methodology": (
            "Gender and age are excluded from the model under ECOA / Regulation B "
            "and are joined back only to measure outcomes. Region IS a model input "
            "and is reported here because coarse geography can act as a proxy."
        ),
    }

    logger.info("measuring the AUC cost of excluding protected attributes (retrains twice)")
    report["cost_of_exclusion"] = cost_of_exclusion()

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "fairness_report.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 74)
    print("FAIR LENDING REPORT")
    print("=" * 74)
    print(f"Decision threshold: PD <= {threshold:.4f}")
    print(f"Overall approval rate: {report['overall_approval_rate']:.2%}\n")
    for attribute, rows in report["groups"].items():
        print(f"--- {attribute} ---")
        table = pd.DataFrame(rows)
        print(
            table[
                [
                    "n",
                    "approval_rate",
                    "mean_pd",
                    "observed_default_rate",
                    "adverse_impact_ratio",
                    "flag",
                ]
            ]
            .set_index(table[attribute])
            .to_string(float_format=lambda v: f"{v:.4f}")
        )
        print()
    cost = report["cost_of_exclusion"]
    print(
        f"Cost of excluding Gender and age: {cost['auc_cost_basis_points']:.1f} bps of ROC-AUC "
        f"({cost['without_protected']:.4f} -> {cost['with_protected']:.4f})"
    )
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
