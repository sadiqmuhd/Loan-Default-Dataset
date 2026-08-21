"""Data contract and quality-profiling tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_default.data.quality import (
    check_categories,
    check_constant_columns,
    check_duplicates,
    check_numeric_ranges,
    check_target_leakage_via_missingness,
    profile,
)
from loan_default.data.schema import (
    allowed_values,
    load_data_contract,
    model_feature_columns,
    numeric_bounds,
    raw_loan_schema,
    validate_raw,
)
from tests.conftest import requires_data

CONTRACT = load_data_contract()


# --------------------------------------------------------------- contract


def test_contract_covers_every_model_feature():
    features = model_feature_columns()
    for column in features["numeric"]:
        assert column in CONTRACT["numeric"]
    for column in features["categorical"]:
        assert column in CONTRACT["categorical"]


def test_allowed_values_reflect_the_dataset():
    assert set(allowed_values("credit_type")) == {"CIB", "CRIF", "EQUI", "EXP"}
    assert "North" in allowed_values("Region")
    assert "north" not in allowed_values("Region")


def test_allowed_values_rejects_unknown_column():
    with pytest.raises(KeyError):
        allowed_values("not_a_column")


def test_numeric_bounds_are_ordered():
    for column in model_feature_columns()["numeric"]:
        low, high = numeric_bounds(column)
        assert low < high


def test_protected_attributes_are_not_model_features():
    features = model_feature_columns()
    everything = features["numeric"] + features["categorical"] + features["engineered"]
    assert "Gender" not in everything
    assert "age" not in everything


# ----------------------------------------------------------- pandera schema


def _valid_frame(n: int = 3) -> pd.DataFrame:
    from loan_default.api.schemas.requests import EXAMPLE_VALUES

    features = model_feature_columns()
    columns = features["numeric"] + features["categorical"]
    return pd.DataFrame([{c: EXAMPLE_VALUES[c] for c in columns} for _ in range(n)])


def test_schema_accepts_valid_records():
    validated = validate_raw(_valid_frame())
    assert len(validated) == 3


def test_schema_rejects_out_of_range_numeric():
    import pandera.errors as pa_errors

    frame = _valid_frame()
    frame.loc[0, "LTV"] = 9999.0
    with pytest.raises(pa_errors.SchemaErrors):
        validate_raw(frame)


def test_schema_rejects_unknown_category():
    import pandera.errors as pa_errors

    frame = _valid_frame()
    frame.loc[0, "credit_type"] = "STD"
    with pytest.raises(pa_errors.SchemaErrors):
        validate_raw(frame)


def test_schema_collects_all_errors_when_lazy():
    import pandera.errors as pa_errors

    frame = _valid_frame()
    frame.loc[0, "LTV"] = 9999.0
    frame.loc[0, "credit_type"] = "STD"
    with pytest.raises(pa_errors.SchemaErrors) as exc:
        validate_raw(frame, lazy=True)
    # Lazy validation must surface both problems, not stop at the first.
    assert len(exc.value.failure_cases) >= 2


def test_schema_is_cached():
    assert raw_loan_schema() is raw_loan_schema()


# --------------------------------------------------------------- quality


def test_duplicate_ids_are_errors():
    frame = pd.DataFrame({"ID": [1, 2, 2, 3]})
    issues = check_duplicates(frame)
    assert any(i.check == "duplicate_id" and i.severity == "error" for i in issues)


def test_duplicate_rows_are_warnings():
    frame = pd.DataFrame({"ID": [1, 2, 3], "x": [1, 1, 2]})
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    assert any(i.check == "duplicate_row" for i in check_duplicates(frame))


def test_clean_frame_has_no_duplicate_issues():
    assert check_duplicates(pd.DataFrame({"ID": [1, 2, 3]})) == []


def test_out_of_range_numerics_are_detected():
    frame = pd.DataFrame({"income": [5000.0, -10.0, 8000.0]})
    issues = check_numeric_ranges(frame, CONTRACT)
    assert issues and issues[0].count == 1


def test_unseen_categories_are_detected():
    frame = pd.DataFrame({"credit_type": ["CIB", "STD", "EXP"]})
    issues = check_categories(frame, CONTRACT)
    assert issues and "STD" in issues[0].examples


def test_constant_columns_are_flagged():
    frame = pd.DataFrame({"year": [2019] * 10, "x": range(10)})
    issues = check_constant_columns(frame)
    assert [i.column for i in issues] == ["year"]


def test_missingness_that_encodes_the_target_is_an_error():
    """The check that would have caught this dataset's central problem."""
    frame = pd.DataFrame(
        {
            "Status": [0] * 50 + [1] * 50,
            "leaky": [1.0] * 50 + [np.nan] * 50,
            "harmless": [1.0] * 95 + [np.nan] * 5,
        }
    )
    issues = check_target_leakage_via_missingness(frame)
    flagged = {i.column for i in issues}
    assert "leaky" in flagged
    assert "harmless" not in flagged


def test_leakage_check_is_silent_without_a_target():
    frame = pd.DataFrame({"a": [1.0, np.nan]})
    assert check_target_leakage_via_missingness(frame) == []


def test_profile_of_a_clean_frame_passes():
    report = profile(_valid_frame(200), CONTRACT, include_leakage_check=False)
    assert report.passed
    assert report.n_rows == 200


@requires_data
def test_profile_flags_the_known_problems_in_the_raw_dataset(raw_data):
    """Documents what is actually wrong with the source file.

    This is a regression test on the data, not the code: if these stop firing,
    the dataset changed and the model card needs revisiting.
    """
    report = profile(raw_data)
    by_column = {(i.column, i.check) for i in report.issues}

    assert ("Interest_rate_spread", "missingness_encodes_target") in by_column
    assert ("rate_of_interest", "missingness_encodes_target") in by_column
    assert ("year", "constant") in by_column
    assert ("income", "out_of_range") in by_column
    assert ("LTV", "out_of_range") in by_column
    assert not report.passed


@requires_data
def test_dataset_has_no_duplicate_ids(raw_data):
    assert not any(i.check == "duplicate_id" for i in check_duplicates(raw_data))
