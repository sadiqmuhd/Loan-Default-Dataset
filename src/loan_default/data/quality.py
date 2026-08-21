"""Data quality profiling.

Distinct from ``schema.py``, which decides whether one record is acceptable and
rejects it if not. This module looks at a whole file and reports what is wrong
with it, including problems that are not grounds for rejection but do change how
you read a model trained on it.

The checks are driven by the same generated contract the API validates against,
so the two cannot disagree about what counts as valid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from loan_default.data.schema import load_data_contract

logger = logging.getLogger(__name__)

Severity = Literal["error", "warning", "info"]


@dataclass
class QualityIssue:
    column: str | None
    check: str
    severity: Severity
    count: int
    detail: str
    examples: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "column": self.column,
            "check": self.check,
            "severity": self.severity,
            "count": self.count,
            "detail": self.detail,
            "examples": self.examples,
        }


@dataclass
class QualityReport:
    n_rows: int
    n_columns: int
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[QualityIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def passed(self) -> bool:
        """Errors block; warnings are for a human to weigh."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "passed": self.passed,
            "n_errors": len(self.errors),
            "n_warnings": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }

    def summary(self) -> str:
        if not self.issues:
            return f"{self.n_rows:,} rows x {self.n_columns} columns: no issues found."
        lines = [
            f"{self.n_rows:,} rows x {self.n_columns} columns: "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        ]
        for issue in sorted(self.issues, key=lambda i: (i.severity != "error", -i.count)):
            marker = {"error": "ERROR", "warning": "WARN ", "info": "INFO "}[issue.severity]
            where = issue.column or "(table)"
            lines.append(f"  {marker} {where:28} {issue.detail}")
        return "\n".join(lines)


def check_duplicates(df: pd.DataFrame, id_column: str = "ID") -> list[QualityIssue]:
    """Duplicate identifiers and fully duplicated rows.

    A repeated loan ID means either a genuine duplicate that would double-count
    exposure, or two different loans sharing an identifier. Both need a human.
    """
    issues: list[QualityIssue] = []
    if id_column in df.columns:
        duplicated = df[id_column].duplicated()
        if (count := int(duplicated.sum())) > 0:
            issues.append(
                QualityIssue(
                    column=id_column,
                    check="duplicate_id",
                    severity="error",
                    count=count,
                    detail=f"{count:,} rows repeat an existing {id_column}",
                    examples=df.loc[duplicated, id_column].head(5).tolist(),
                )
            )

    if (count := int(df.duplicated().sum())) > 0:
        issues.append(
            QualityIssue(
                column=None,
                check="duplicate_row",
                severity="warning",
                count=count,
                detail=f"{count:,} rows are exact duplicates of another row",
            )
        )
    return issues


def check_numeric_ranges(df: pd.DataFrame, contract: dict[str, Any]) -> list[QualityIssue]:
    """Values outside the documented domain, e.g. negative income, LTV of 7831%."""
    issues: list[QualityIssue] = []
    for column, spec in contract.get("numeric", {}).items():
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        low, high = float(spec["min"]), float(spec["max"])
        out_of_range = (series < low) | (series > high)
        if (count := int(out_of_range.sum())) > 0:
            issues.append(
                QualityIssue(
                    column=column,
                    check="out_of_range",
                    severity="error",
                    count=count,
                    detail=(
                        f"{count:,} values outside [{low:,.0f}, {high:,.0f}] "
                        f"(observed {series.min():,.1f} to {series.max():,.1f})"
                    ),
                    examples=series[out_of_range].head(5).tolist(),
                )
            )
    return issues


def check_categories(df: pd.DataFrame, contract: dict[str, Any]) -> list[QualityIssue]:
    """Categorical levels the contract does not know about."""
    issues: list[QualityIssue] = []
    for column, spec in contract.get("categorical", {}).items():
        if column not in df.columns:
            continue
        unseen = set(df[column].dropna().astype(str).unique()) - set(spec["allowed"])
        if unseen:
            affected = int(df[column].astype(str).isin(unseen).sum())
            issues.append(
                QualityIssue(
                    column=column,
                    check="unseen_category",
                    severity="error",
                    count=affected,
                    detail=f"{affected:,} rows use levels absent from the contract",
                    examples=sorted(unseen)[:5],
                )
            )
    return issues


def check_missingness(df: pd.DataFrame, threshold: float = 0.05) -> list[QualityIssue]:
    """Columns missing more than ``threshold`` of their values."""
    issues: list[QualityIssue] = []
    for column in df.columns:
        rate = float(df[column].isna().mean())
        if rate > threshold:
            issues.append(
                QualityIssue(
                    column=column,
                    check="high_missingness",
                    severity="warning",
                    count=int(df[column].isna().sum()),
                    detail=f"{rate:.1%} missing",
                )
            )
    return issues


def check_target_leakage_via_missingness(
    df: pd.DataFrame, target: str = "Status", threshold: float = 0.90
) -> list[QualityIssue]:
    """Columns whose missingness pattern all but reproduces the target.

    This is the check that would have caught this dataset's central problem
    before any model was fitted: ``Interest_rate_spread.isna()`` matches
    ``Status`` on every one of the 148,670 rows. Any column scoring above the
    threshold must be excluded from the feature set, not merely noted.
    """
    issues: list[QualityIssue] = []
    if target not in df.columns:
        return issues

    y = df[target]
    for column in df.columns:
        if column == target or not df[column].isna().any():
            continue
        agreement = float((df[column].isna().astype(int) == y).mean())
        # Consider the complement too: missingness may encode the inverse.
        agreement = max(agreement, 1.0 - agreement)
        if agreement >= threshold:
            issues.append(
                QualityIssue(
                    column=column,
                    check="missingness_encodes_target",
                    severity="error",
                    count=int(df[column].isna().sum()),
                    detail=(
                        f"missingness matches {target} on {agreement:.3%} of rows - "
                        "exclude from features"
                    ),
                )
            )
    return issues


def check_constant_columns(df: pd.DataFrame) -> list[QualityIssue]:
    """Zero-variance columns carry no signal and should not be features."""
    issues: list[QualityIssue] = []
    for column in df.columns:
        values = df[column].dropna()
        if len(values) > 0 and values.nunique() == 1:
            issues.append(
                QualityIssue(
                    column=column,
                    check="constant",
                    severity="warning",
                    count=len(values),
                    detail=f"single value throughout ({values.iloc[0]!r})",
                )
            )
    return issues


def check_outliers(df: pd.DataFrame, columns: list[str] | None = None) -> list[QualityIssue]:
    """Extreme values by the interquartile rule, reported for attention only.

    An outlier is not necessarily an error - large mortgages exist - so these
    are informational and never block.
    """
    issues: list[QualityIssue] = []
    numeric = columns or df.select_dtypes(include=np.number).columns.tolist()
    for column in numeric:
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) < 100:
            continue
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr <= 0:
            continue
        extreme = (series < q1 - 3 * iqr) | (series > q3 + 3 * iqr)
        if (count := int(extreme.sum())) > 0 and count / len(series) > 0.001:
            issues.append(
                QualityIssue(
                    column=column,
                    check="extreme_values",
                    severity="info",
                    count=count,
                    detail=f"{count:,} values beyond 3x IQR ({count / len(series):.2%})",
                    examples=series[extreme].head(3).tolist(),
                )
            )
    return issues


def profile(
    df: pd.DataFrame,
    contract: dict[str, Any] | None = None,
    *,
    target: str = "Status",
    include_leakage_check: bool = True,
) -> QualityReport:
    """Run every check and collect the findings."""
    contract = contract or load_data_contract()
    issues: list[QualityIssue] = []
    issues += check_duplicates(df)
    issues += check_numeric_ranges(df, contract)
    issues += check_categories(df, contract)
    issues += check_missingness(df)
    issues += check_constant_columns(df)
    issues += check_outliers(df)
    if include_leakage_check:
        issues += check_target_leakage_via_missingness(df, target)

    report = QualityReport(n_rows=len(df), n_columns=df.shape[1], issues=issues)
    logger.info(
        "quality profile: %d rows, %d errors, %d warnings",
        report.n_rows,
        len(report.errors),
        len(report.warnings),
    )
    return report
