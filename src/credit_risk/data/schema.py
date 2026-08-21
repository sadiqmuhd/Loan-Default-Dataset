"""Data validation schema, built from the generated data contract.

This is the single source of truth for what a valid loan record looks like.
The training pipeline and the API request models are both derived from it, so
the two cannot drift apart - which is the failure the original codebase had
(the hand-written API schema rejected 99.62% of the training data).

Regenerate the underlying contract with:
    python scripts/generate_data_contract.py
"""

from __future__ import annotations

import functools
from typing import Any

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

from credit_risk.config import CONFIG_DIR, _load_yaml, load_model_config


@functools.lru_cache(maxsize=1)
def load_data_contract() -> dict[str, Any]:
    """The generated contract: allowed categorical levels and numeric bounds."""
    return _load_yaml(CONFIG_DIR / "data_contract.yaml")


def allowed_values(column: str) -> list[str]:
    """Allowed levels for a categorical column, straight from the data."""
    contract = load_data_contract()
    if column not in contract["categorical"]:
        raise KeyError(f"{column!r} is not a categorical column in the data contract")
    return list(contract["categorical"][column]["allowed"])


def numeric_bounds(column: str) -> tuple[float, float]:
    """Inclusive (min, max) bounds for a numeric column."""
    contract = load_data_contract()
    if column not in contract["numeric"]:
        raise KeyError(f"{column!r} is not a numeric column in the data contract")
    spec = contract["numeric"][column]
    return float(spec["min"]), float(spec["max"])


def model_feature_columns() -> dict[str, list[str]]:
    """The columns the model actually consumes, split by kind."""
    cfg = load_model_config()
    return {
        "numeric": list(cfg["features"]["numeric"]),
        "categorical": list(cfg["features"]["categorical"]),
        "engineered": list(cfg["features"]["engineered"]),
    }


@functools.lru_cache(maxsize=1)
def raw_loan_schema(strict: bool = False) -> DataFrameSchema:
    """Pandera schema for a raw loan record, before feature engineering.

    Only covers the columns the model consumes. Columns excluded from the model
    (leakage, protected characteristics, identifiers) are intentionally absent -
    they are not part of the serving contract at all.
    """
    contract = load_data_contract()
    features = model_feature_columns()
    columns: dict[str, Column] = {}

    for col in features["numeric"]:
        spec = contract["numeric"][col]
        columns[col] = Column(
            float,
            checks=[
                pa.Check.ge(float(spec["min"])),
                pa.Check.le(float(spec["max"])),
            ],
            nullable=bool(spec["nullable"]),
            coerce=True,
            required=True,
            description=f"{col} in [{spec['min']}, {spec['max']}]",
        )

    for col in features["categorical"]:
        spec = contract["categorical"][col]
        columns[col] = Column(
            str,
            checks=[pa.Check.isin(list(spec["allowed"]))],
            nullable=bool(spec["nullable"]),
            coerce=True,
            required=True,
            description=f"{col} in {spec['allowed']}",
        )

    return DataFrameSchema(
        columns=columns,
        strict=strict,
        coerce=True,
        name="RawLoanRecord",
        description="A loan application as consumed by the PD model.",
    )


def validate_raw(df, *, lazy: bool = True):
    """Validate a raw loan DataFrame. Raises ``pa.errors.SchemaErrors`` on failure.

    ``lazy=True`` collects every violation rather than stopping at the first,
    which is what the API needs in order to return a complete error payload.
    """
    return raw_loan_schema().validate(df, lazy=lazy)
