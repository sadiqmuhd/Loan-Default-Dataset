"""API response models.

Every assessment carries the model version, the request id and the assumptions
behind its loss figures, so any decision the service made can be reconstructed
later from the response alone.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ReasonCodeOut(BaseModel):
    feature: str = Field(description="Underlying feature the reason relates to.")
    label: str = Field(description="Human-readable description.")
    contribution: float = Field(description="Signed SHAP contribution, log-odds space.")
    direction: Literal["increases_risk", "reduces_risk"]
    value: Any = Field(default=None, description="The applicant's value for this feature.")


class ExplanationOut(BaseModel):
    reason_codes: list[str] = Field(
        default_factory=list,
        description=(
            "Compact adverse-action style codes for the principal risk factors, "
            "strongest first. Derived from the SHAP drivers and the applicant's "
            "actual values."
        ),
        examples=[["HIGH_DTI", "HIGH_LTV", "LARGE_EXPOSURE"]],
    )
    risk_drivers: list[ReasonCodeOut] = Field(description="Factors increasing assessed risk.")
    risk_reducers: list[ReasonCodeOut] = Field(description="Factors reducing assessed risk.")
    base_value: float = Field(description="Model base score before feature contributions.")
    method: str
    note: str


class LossComponentsOut(BaseModel):
    pd: float = Field(description="Calibrated probability of default.")
    lgd: float = Field(description="Loss given default. A COLLATERAL PROXY, not a model.")
    ead: float = Field(description="Exposure at default (origination amount).")
    expected_loss: float = Field(description="EL = PD x LGD x EAD, in currency units.")
    expected_loss_rate: float = Field(description="EL as a fraction of exposure.")
    collateral_value: float | None = None
    lgd_method: str


class DecisionOut(BaseModel):
    decision: Literal["APPROVE", "REVIEW", "DECLINE"]
    reason: str = Field(description="Plain-language justification for the decision.")
    break_even_pd: float = Field(description="PD at which expected margin equals expected loss.")
    expected_profit: float
    expected_revenue: float
    expected_loss: float


class RiskAssessmentResponse(BaseModel):
    """The full credit risk assessment for one application."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "request_id": "3f8a1c2e-5b7d-4a91-8e6f-1d2c3b4a5e6f",
                "model_version": "v20260821T004512Z",
                "probability_of_default": 0.0363,
                "risk_grade": "B",
                "grade_description": "Low risk",
                "decision": {
                    "decision": "APPROVE",
                    "reason": "PD 3.63% is below the break-even PD of 36.33%.",
                    "break_even_pd": 0.3633,
                    "expected_profit": 423_711.0,
                    "expected_revenue": 427_600.0,
                    "expected_loss": 3_889.0,
                },
            }
        }
    )

    request_id: str = Field(description="Correlation id, echoed in logs.")
    model_version: str = Field(description="Exact model artifact that produced this.")
    assessed_at: str = Field(description="UTC timestamp of the assessment.")

    probability_of_default: float = Field(ge=0.0, le=1.0, description="Calibrated PD.")
    risk_grade: str
    grade_description: str

    loss: LossComponentsOut
    decision: DecisionOut
    explanation: ExplanationOut | None = None

    assumptions_version: str = Field(
        description=(
            "Version of config/risk_policy.yaml in force for this assessment. "
            "Loss figures are only comparable across responses sharing it."
        )
    )
    assumptions: dict[str, Any] = Field(
        description="Assumptions behind the LGD/EAD/EL figures and the decision cut-off."
    )
    latency_ms: float | None = None


class BatchItemResult(BaseModel):
    """One row of a batch response. Either an assessment or a per-row error."""

    index: int
    assessment: RiskAssessmentResponse | None = None
    error: str | None = None


class BatchAssessmentResponse(BaseModel):
    request_id: str
    model_version: str
    n_submitted: int
    n_succeeded: int
    n_failed: int
    results: list[BatchItemResult]
    portfolio: dict[str, Any] | None = Field(
        default=None, description="Aggregate metrics across successfully scored rows."
    )
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    service: str
    api_version: str


class ReadinessResponse(BaseModel):
    """Readiness verifies the model scores, not merely that an artifact exists."""

    status: Literal["ready", "not_ready"]
    model_loaded: bool
    model_version: str | None = None
    canary_prediction_ok: bool = Field(
        description="Whether a known-good record scored successfully at startup."
    )
    detail: str | None = None


class ModelMetadataResponse(BaseModel):
    model_version: str
    model_type: str
    trained_at: str
    data_sha256: str
    n_training_rows: int
    default_rate: float
    feature_columns: list[str]
    excluded_columns: dict[str, list[str]]
    calibration_method: str
    seed: int
    library_versions: dict[str, str]
    assumptions: list[str]
    limitations: list[str]


class ModelMetricsResponse(BaseModel):
    model_version: str
    calibrated: dict[str, Any]
    uncalibrated: dict[str, Any]
    candidate_cv_pr_auc: dict[str, float]
    data_provenance: dict[str, Any]


class ErrorDetail(BaseModel):
    field: str | None = None
    message: str


class ErrorResponse(BaseModel):
    """RFC-7807-flavoured error body. Never leaks internal exception text."""

    error: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Safe human-readable summary.")
    request_id: str
    details: list[ErrorDetail] = Field(default_factory=list)
