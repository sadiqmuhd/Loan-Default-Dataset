"""Schema round-trip tests.

The original hand-written Pydantic schema rejected 148,111 of 148,670 real loan
records - 99.62% of the dataset the model was trained on - while permitting
eleven enum values that appear nowhere in the data ('nf', 'l3', 'STD', 'OTH',
'NONE', 'to_bank', 'north', 'east', 'west', 'indirect', 'other').

These tests make that class of failure structurally impossible: the request
model is generated from the data contract, and real rows must validate.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from loan_default.api.schemas.requests import LoanApplication
from loan_default.data.schema import load_data_contract, model_feature_columns
from tests.conftest import requires_data

SAMPLE_SIZE = 2000


def _to_payload(row) -> dict:
    """Convert a raw dataframe row to an API payload, dropping NaN numerics."""
    features = model_feature_columns()
    payload = {}
    for column in features["numeric"] + features["categorical"]:
        value = row.get(column)
        if value is None or (isinstance(value, float) and np.isnan(value)):
            continue
        payload[column] = value
    return payload


@requires_data
def test_real_dataset_rows_validate(raw_data):
    """The headline regression test: real records must be acceptable."""
    contract = load_data_contract()
    features = model_feature_columns()

    # Restrict to rows that are in-domain for the model: complete on the
    # required numerics and within the documented bounds.
    frame = raw_data.dropna(subset=features["numeric"])
    for column in features["numeric"]:
        spec = contract["numeric"][column]
        frame = frame[(frame[column] >= spec["min"]) & (frame[column] <= spec["max"])]

    assert len(frame) > 100_000, (
        f"Only {len(frame):,} rows are in-domain; the contract may be too strict."
    )

    sample = frame.sample(n=min(SAMPLE_SIZE, len(frame)), random_state=42)
    failures = []
    for _, row in sample.iterrows():
        try:
            LoanApplication(**_to_payload(row))
        except ValidationError as exc:
            failures.append((row.get("ID"), exc.errors()[:2]))

    acceptance = 1 - len(failures) / len(sample)
    assert not failures, (
        f"{len(failures)} of {len(sample)} real rows were rejected "
        f"(acceptance {acceptance:.2%}). First failures: {failures[:3]}"
    )


@requires_data
def test_every_categorical_level_is_accepted(raw_data):
    """Every level that occurs in the data must be a legal input."""
    contract = load_data_contract()
    for column, spec in contract["categorical"].items():
        observed = set(raw_data[column].dropna().unique())
        allowed = set(spec["allowed"])
        missing = observed - allowed
        assert not missing, f"{column}: real levels rejected by the contract: {missing}"


@requires_data
def test_contract_contains_no_phantom_levels(raw_data):
    """No allowed value may be absent from the data.

    This is what caught the original schema out: it allowed 'STD' and 'OTH' for
    credit_type when the real levels are CIB, CRIF, EQUI and EXP.
    """
    contract = load_data_contract()
    for column, spec in contract["categorical"].items():
        observed = {str(v) for v in raw_data[column].dropna().unique()}
        phantom = set(spec["allowed"]) - observed
        assert not phantom, f"{column}: contract allows values absent from the data: {phantom}"


def test_protected_attributes_are_not_accepted(example_application):
    """Gender and age must be rejected outright, not silently ignored.

    ``extra="forbid"`` means the service cannot receive a protected
    characteristic even by accident.
    """
    for attribute, value in (("Gender", "Male"), ("age", "35-44")):
        with pytest.raises(ValidationError):
            LoanApplication(**{**example_application, attribute: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("loan_amount", -1000.0),
        ("loan_amount", 0.0),
        ("income", -5.0),
        ("LTV", 7831.25),  # the real out-of-domain value in the dataset
        ("dtir1", 150.0),
        ("Credit_Score", 99.0),
        ("term", 0.0),
    ],
)
def test_out_of_domain_numerics_rejected(example_application, field, value):
    """The original schema had no numeric bounds at all."""
    with pytest.raises(ValidationError):
        LoanApplication(**{**example_application, field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("credit_type", "STD"),  # allowed by the original schema, absent from data
        ("Region", "north"),  # original used lowercase; real value is "North"
        ("loan_purpose", "p9"),
        ("submission_of_application", "to_bank"),  # original allowed this; not real
    ],
)
def test_invalid_categories_rejected(example_application, field, value):
    with pytest.raises(ValidationError):
        LoanApplication(**{**example_application, field: value})


def test_example_payload_is_valid(example_application):
    """The documented OpenAPI example must actually work."""
    assert LoanApplication(**example_application) is not None
