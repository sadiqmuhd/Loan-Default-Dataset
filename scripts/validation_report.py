"""Generate the model validation report.

Writes reports/validation_report.md from the committed artifact: grade
rank-ordering, calibration, and performance by portfolio segment. Regenerating
it after every retrain keeps the documentation honest, since the numbers come
from the model rather than from memory.

    python scripts/validation_report.py
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loan_default.config import get_settings, load_model_config  # noqa: E402
from loan_default.data.loader import load_dataset  # noqa: E402
from loan_default.logging_config import configure_logging  # noqa: E402
from loan_default.models.calibration import reliability_table  # noqa: E402
from loan_default.models.evaluate import confusion_at_threshold, evaluate  # noqa: E402
from loan_default.models.registry import ModelRegistry  # noqa: E402
from loan_default.models.segments import segment_report  # noqa: E402
from loan_default.risk.grades import grade_summary, is_monotonic  # noqa: E402
from loan_default.risk.policy import break_even_pd  # noqa: E402

logger = logging.getLogger(__name__)


def _table(df: pd.DataFrame, floats: str = "{:.4f}") -> str:
    """Render a frame as a markdown table, keeping counts as integers."""
    formatted = df.copy()
    for column in formatted.select_dtypes("number").columns:
        series = formatted[column]
        if column in {"n", "count"} or (series.dropna() % 1 == 0).all():
            formatted[column] = series.map(lambda v: f"{int(v):,}")
        else:
            formatted[column] = series.map(lambda v: floats.format(v))
    header = "| " + " | ".join(str(c) for c in formatted.columns) + " |"
    divider = "|" + "|".join("---" for _ in formatted.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in formatted.values]
    return "\n".join([header, divider, *rows])


def main() -> int:
    configure_logging()
    settings = get_settings()
    cfg = load_model_config()

    model, metadata, metrics = ModelRegistry(settings.artifacts_dir).load("latest")
    dataset = load_dataset()
    _, X_test, _, y_test = train_test_split(
        dataset.X,
        dataset.y,
        test_size=cfg["validation"]["test_size"],
        stratify=dataset.y,
        random_state=cfg["seed"],
    )
    frame = X_test[metadata.feature_columns]
    y_prob = model.predict_proba(frame)[:, 1]

    report = evaluate(y_test, y_prob)
    grades = grade_summary(y_prob, y_test)
    reliability = reliability_table(y_test, y_prob)
    segments = segment_report(frame, y_test, y_prob)
    threshold = break_even_pd(0.45)
    confusion = confusion_at_threshold(y_test, y_prob, threshold)

    parts: list[str] = []
    parts.append(
        f"""# Model Validation Report

Generated {datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")} from model
`{metadata.model_version}` on the held-out test set ({report.n:,} loans,
{report.base_rate:.2%} default rate).

Regenerate with `python scripts/validation_report.py`.

## 1. Headline performance

| Metric | Value |
|---|---|
| ROC-AUC | {report.roc_auc:.4f} |
| Gini | {report.gini:.4f} |
| PR-AUC | {report.pr_auc:.4f} (baseline {report.base_rate:.4f}) |
| KS | {report.ks_statistic:.4f} |
| Brier score | {report.brier_score:.4f} |
| Mean predicted PD | {report.mean_predicted_pd:.4f} |
| Observed default rate | {report.base_rate:.4f} |
| Mean calibration error | {report.calibration_error:.4f} |
| Max calibration error | {report.max_calibration_error:.4f} |
"""
    )

    monotonic = is_monotonic(grades)
    parts.append(
        f"""## 2. Risk grade rank ordering

Observed default rate must increase from A through G, or the grades cannot
support a lending decision.

**Monotonic: {"yes" if monotonic else "NO - INVESTIGATE"}**

{_table(grades[["grade", "n", "share", "mean_pd", "observed_default_rate"]])}
"""
    )

    parts.append(
        f"""## 3. Calibration

Predicted versus observed default rate by predicted-PD decile. A well
calibrated model tracks the diagonal; see `reports/reliability_curve.png`.

{_table(reliability[["n", "predicted", "observed", "gap"]])}
"""
    )

    comparison = metrics.get("calibration_comparison")
    if comparison:
        # Each entry already carries its own "method" key, so reset the index away.
        table = pd.DataFrame(comparison).T.reset_index(drop=True)
        parts.append(
            f"""### Calibration method selection

Both candidates were fitted on the held-out calibration slice and measured on
the test set. Selection is on calibration error rather than Brier score, which
mixes discrimination and calibration together and barely moves when only the
latter improves.

{_table(table[["method", "brier_score", "calibration_error", "max_calibration_error", "roc_auc"]])}
"""
        )

    parts.append(
        f"""## 4. Decision outcomes

At the break-even threshold for an unsecured-equivalent LGD of 0.45
(PD <= {threshold:.4f}):

| Quantity | Value |
|---|---|
| Approval rate | {confusion["approval_rate"]:.2%} |
| Default rate among approved | {confusion["bad_rate_in_approved"]:.2%} |
| Recall on defaulters | {confusion["recall"]:.2%} |
| Precision | {confusion["precision"]:.2%} |
| True negatives | {confusion["true_negatives"]:,} |
| False positives | {confusion["false_positives"]:,} |
| False negatives | {confusion["false_negatives"]:,} |
| True positives | {confusion["true_positives"]:,} |
"""
    )

    parts.append("## 5. Performance by segment\n")
    parts.append(
        "A single headline AUC can hide a segment where the model does not "
        "discriminate. Segments below "
        f"{segments['min_segment_size']} loans are not evaluated, since the "
        "estimate would be too noisy to act on.\n"
    )
    for dimension, rows in segments["dimensions"].items():
        table = pd.DataFrame(rows)
        parts.append(f"### {dimension}\n")
        parts.append(
            _table(table[["segment", "n", "default_rate", "mean_pd", "calibration_gap", "roc_auc"]])
            + "\n"
        )

    if segments["flagged"]:
        parts.append(f"### Flagged for review ({segments['n_flagged']})\n")
        for item in segments["flagged"]:
            parts.append(
                f"- **{item['dimension']} = {item['segment']}** (n={item['n']:,}): "
                + "; ".join(item["reasons"])
            )
        parts.append("")
    else:
        parts.append(
            f"### Review triggers\n\nNo segment fell below the ROC-AUC floor of "
            f"{segments['auc_floor']} or exceeded the calibration tolerance of "
            f"{segments['calibration_tolerance']}.\n"
        )

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / "validation_report.md"
    out.write_text("\n".join(parts), encoding="utf-8")

    print(f"Wrote {out}")
    print(f"  grades monotonic: {monotonic}")
    print(f"  segments flagged: {segments['n_flagged']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
