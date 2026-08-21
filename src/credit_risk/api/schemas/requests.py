"""API request models, GENERATED FROM THE DATA CONTRACT.

This is the fix for the original codebase's most damaging defect: the
hand-written Pydantic schema rejected 148,111 of 148,670 real loan records
(99.62%) while permitting eleven enum values that appear nowhere in the data.

Here the enums and bounds are built at import time from
``config/data_contract.yaml``, which is itself generated from the dataset. The
API contract therefore cannot drift from the data.
``tests/api/test_schema_roundtrip.py`` asserts that real rows validate.

Protected characteristics (``Gender``, ``age``) are deliberately absent from the
request model. The service does not merely ignore them - it never receives them,
which is the strongest position under ECOA / Regulation B.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model

from credit_risk.data.schema import load_data_contract, model_feature_columns

# Human-readable descriptions for the OpenAPI docs.
FIELD_DESCRIPTIONS: dict[str, str] = {
    "loan_amount": "Requested loan amount, in currency units.",
    "term": "Loan term in months.",
    "property_value": "Appraised value of the property securing the loan.",
    "income": "Applicant income. MONTHLY, not annual - see MODEL_CARD.md.",
    "Credit_Score": (
        "Applicant credit score (300-900). NOTE: this field is not predictive in "
        "the training data (univariate ROC-AUC 0.503) and appears to be randomly "
        "generated; it is retained for contract completeness."
    ),
    "LTV": "Loan-to-value ratio, as a percentage.",
    "dtir1": "Debt-to-income ratio, as a percentage.",
    "loan_limit": "Whether the loan conforms to the applicable lending limit.",
    "loan_purpose": "Purpose code for the loan.",
    "lump_sum_payment": "Whether the loan carries a balloon / lump-sum repayment.",
    "Neg_ammortization": "Whether the loan permits negative amortisation.",
    "interest_only": "Whether the loan is interest-only.",
    "occupancy_type": "Primary residence, secondary residence or investment.",
    "business_or_commercial": "Whether the loan is for business or commercial purposes.",
    "submission_of_application": "Channel through which the application was submitted.",
    "Region": "Coarse geographic region.",
}

EXAMPLE_VALUES: dict[str, Any] = {
    "loan_amount": 296500.0,
    "term": 360.0,
    "property_value": 418000.0,
    "income": 5760.0,
    "Credit_Score": 699,
    "LTV": 70.9,
    "dtir1": 39.0,
    "loan_limit": "cf",
    "approv_in_adv": "nopre",
    "loan_type": "type1",
    "loan_purpose": "p3",
    "Credit_Worthiness": "l1",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Secured_by": "home",
    "total_units": "1U",
    "credit_type": "CIB",
    "co-applicant_credit_type": "CIB",
    "submission_of_application": "to_inst",
    "Region": "North",
    "Security_Type": "direct",
}


def _build_loan_application_model() -> type[BaseModel]:
    """Construct the request model from the generated data contract."""
    contract = load_data_contract()
    features = model_feature_columns()
    fields: dict[str, Any] = {}

    for column in features["numeric"]:
        spec = contract["numeric"][column]
        annotation = float
        field = Field(
            ...,
            ge=float(spec["min"]),
            le=float(spec["max"]),
            description=FIELD_DESCRIPTIONS.get(column, column),
        )
        fields[column] = (annotation, field)

    for column in features["categorical"]:
        spec = contract["categorical"][column]
        # Literal gives a proper enum in the OpenAPI schema.
        annotation = Literal[tuple(spec["allowed"])]  # type: ignore[valid-type]
        if spec["nullable"]:
            annotation = annotation | None  # type: ignore[assignment]

        alias = column if "-" not in column else column
        py_name = column.replace("-", "_")
        field = Field(
            default=None if spec["nullable"] else ...,
            alias=alias,
            description=FIELD_DESCRIPTIONS.get(column, f"One of {spec['allowed']}"),
        )
        fields[py_name] = (annotation, field)

    model = create_model(
        "LoanApplication",
        __config__=ConfigDict(
            populate_by_name=True,
            extra="forbid",
            json_schema_extra={"example": EXAMPLE_VALUES},
        ),
        **fields,
    )
    model.__doc__ = (
        "A loan application to be assessed.\n\n"
        "Field enumerations and numeric bounds are generated from the training "
        "dataset (config/data_contract.yaml). Protected characteristics are not "
        "collected."
    )
    return model


LoanApplication = _build_loan_application_model()


class BatchAssessmentRequest(BaseModel):
    """A batch of applications to score in one call."""

    model_config = ConfigDict(extra="forbid")

    applications: Annotated[
        list[LoanApplication],  # type: ignore[valid-type]
        Field(min_length=1, description="Applications to assess."),
    ]
    include_explanations: bool = Field(
        default=False,
        description=(
            "Compute SHAP reason codes for every row. Materially slower; leave "
            "off for bulk portfolio scoring."
        ),
    )


class StressTestRequest(BaseModel):
    """Run the configured stress scenarios over a sample of the portfolio."""

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(
        default=5000,
        ge=100,
        le=50_000,
        description="Rows sampled from the held-out portfolio for the exercise.",
    )
    scenarios: list[str] | None = Field(
        default=None,
        description="Scenario names to run. Defaults to all configured scenarios.",
    )
    include_sensitivity: bool = Field(
        default=False,
        description="Also run single-variable sensitivity sweeps (tornado data).",
    )
