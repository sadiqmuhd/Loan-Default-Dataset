"""Reason code derivation.

A code must never contradict the application it describes. Emitting HIGH_DTI on
a file with a 20% debt-to-income ratio would be worse than emitting nothing,
because a reason code is the part of the output a human is expected to act on.
"""

from __future__ import annotations

import pytest

from loan_default.models.explain import ReasonCode, derive_reason_codes


def driver(feature: str, value, contribution: float = 0.5) -> ReasonCode:
    return ReasonCode(
        feature=feature,
        label=feature,
        contribution=contribution,
        direction="increases_risk",
        value=value,
    )


def test_high_dti_fires_only_when_dti_is_actually_high():
    assert derive_reason_codes([driver("dtir1", 58.0)]) == ["HIGH_DTI"]
    assert derive_reason_codes([driver("dtir1", 20.0)]) == []


def test_high_ltv_respects_its_threshold():
    assert derive_reason_codes([driver("LTV", 95.0)]) == ["HIGH_LTV"]
    assert derive_reason_codes([driver("LTV", 60.0)]) == []


@pytest.mark.parametrize(
    ("feature", "adverse", "benign", "code"),
    [
        ("Neg_ammortization", "neg_amm", "not_neg", "NEGATIVE_AMORTISATION"),
        ("lump_sum_payment", "lpsm", "not_lpsm", "BALLOON_REPAYMENT"),
        ("interest_only", "int_only", "not_int", "INTEREST_ONLY_STRUCTURE"),
        ("occupancy_type", "ir", "pr", "NON_PRIMARY_RESIDENCE"),
        ("business_or_commercial", "b/c", "nob/c", "COMMERCIAL_PURPOSE"),
    ],
)
def test_categorical_codes_match_real_contract_values(feature, adverse, benign, code):
    """The enum values are the ones in the data, not ones that seemed plausible.

    An earlier version tested `Neg_ammortization == "neg"`; the dataset uses
    `neg_amm`, so the rule could never have fired.
    """
    assert derive_reason_codes([driver(feature, adverse)]) == [code]
    assert derive_reason_codes([driver(feature, benign)]) == []


def test_codes_are_ordered_by_contribution_and_capped():
    drivers = [
        driver("dtir1", 58.0, 0.9),
        driver("LTV", 95.0, 0.7),
        driver("lump_sum_payment", "lpsm", 0.5),
        driver("interest_only", "int_only", 0.3),
        driver("occupancy_type", "ir", 0.1),
    ]
    codes = derive_reason_codes(drivers, limit=4)
    assert codes == ["HIGH_DTI", "HIGH_LTV", "BALLOON_REPAYMENT", "INTEREST_ONLY_STRUCTURE"]


def test_unknown_features_and_none_values_are_skipped_quietly():
    assert derive_reason_codes([driver("not_a_feature", 1.0)]) == []
    assert derive_reason_codes([driver("dtir1", None)]) == []


def test_no_duplicate_codes():
    assert derive_reason_codes([driver("dtir1", 58.0), driver("dtir1", 60.0)]) == ["HIGH_DTI"]


def test_every_rule_names_a_real_feature_or_engineered_column():
    """Guards against a rule silently referring to a column that does not exist."""
    from loan_default.data.schema import load_data_contract
    from loan_default.models.explain import _CODE_RULES

    contract = load_data_contract()
    known = set(contract["numeric"]) | set(contract["categorical"])
    engineered = {
        "loan_to_income",
        "property_to_income",
        "loan_to_value_ratio",
        "payment_to_income",
    }

    for feature, code, _ in _CODE_RULES:
        assert feature in known or feature in engineered, (
            f"Rule {code} refers to {feature!r}, which is not in the data contract"
        )
