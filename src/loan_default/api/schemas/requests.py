"""Request models, generated from the data contract.

Enumerations and numeric bounds are built at import time from
``config/data_contract.yaml``, which is itself derived from the dataset. Writing
them by hand invites a schema that rejects records the model was trained on, so
they are generated instead and ``test_schema_roundtrip`` checks real rows still
validate.

Gender and age are absent from the request model by design. The service does not
ignore them, it never receives them - which is a stronger fair-lending position
than filtering after the fact.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from loan_default.data.schema import load_data_contract, model_feature_columns

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
    annotation: Any

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
        annotation = Literal[tuple(spec["allowed"])]
        if spec["nullable"]:
            annotation = annotation | None

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
    """A batch of applications to score in one call.

    Applications are typed as raw mappings rather than ``list[LoanApplication]``
    on purpose. Pydantic validates a typed list eagerly and rejects the whole
    request if any element fails, which for a nightly portfolio run means one bad
    row costs you the other 4,999. Each row is validated individually in the
    router instead, so callers get per-row errors alongside successful scores.
    """

    model_config = ConfigDict(extra="forbid")

    applications: Annotated[
        list[dict[str, Any]],
        Field(min_length=1, description="Applications to assess."),
    ]
    include_explanations: bool = Field(
        default=False,
        description=(
            "Compute SHAP reason codes for every row. Materially slower; leave "
            "off for bulk portfolio scoring."
        ),
    )


def validate_application(payload: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Validate one application, returning ``(cleaned, None)`` or ``(None, error)``.

    Used by the batch endpoint so a single malformed row does not fail the batch.
    """
    try:
        model = LoanApplication(**payload)
    except ValidationError as exc:
        problems = [f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()]
        return None, "; ".join(problems[:5])
    return model.model_dump(by_alias=True), None


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
