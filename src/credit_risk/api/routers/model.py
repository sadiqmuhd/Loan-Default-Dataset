"""Model governance endpoints.

These are the traceability surface: given a `model_version` from any assessment
response, a reviewer can retrieve the exact training data hash, the feature
contract, the accepted metrics, and the documented assumptions and limitations.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from credit_risk.api.dependencies import get_scoring_service
from credit_risk.api.schemas.responses import (
    ErrorResponse,
    ModelMetadataResponse,
    ModelMetricsResponse,
)
from credit_risk.api.service import ScoringService
from credit_risk.risk.expected_loss import assumption_disclosure
from credit_risk.risk.grades import grade_scale
from credit_risk.risk.policy import policy_disclosure

router = APIRouter(prefix="/v1/model", tags=["model governance"])

ERRORS = {503: {"model": ErrorResponse, "description": "Model not loaded."}}


@router.get(
    "/metadata",
    response_model=ModelMetadataResponse,
    summary="Model provenance and governance metadata",
    description=(
        "Version, training timestamp, SHA-256 of the training data, the full "
        "feature contract including what was excluded and why, plus documented "
        "assumptions and limitations."
    ),
    responses=ERRORS,
)
def metadata(service: ScoringService = Depends(get_scoring_service)) -> ModelMetadataResponse:
    meta = service.metadata
    return ModelMetadataResponse(
        model_version=meta.model_version,
        model_type=meta.model_type,
        trained_at=meta.trained_at,
        data_sha256=meta.data_sha256,
        n_training_rows=meta.n_training_rows,
        default_rate=meta.default_rate,
        feature_columns=meta.feature_columns,
        excluded_columns=meta.excluded_columns,
        calibration_method=meta.calibration_method,
        seed=meta.seed,
        library_versions=meta.library_versions,
        assumptions=meta.assumptions,
        limitations=meta.limitations,
    )


@router.get(
    "/metrics",
    response_model=ModelMetricsResponse,
    summary="Accepted model performance metrics",
    description=(
        "Discrimination and calibration metrics measured on the held-out test "
        "set at training time, for both the calibrated and uncalibrated model, "
        "plus the cross-validated scores of every candidate considered."
    ),
    responses=ERRORS,
)
def metrics(service: ScoringService = Depends(get_scoring_service)) -> ModelMetricsResponse:
    payload = service.metrics
    return ModelMetricsResponse(
        model_version=service.metadata.model_version,
        calibrated=payload.get("calibrated", {}),
        uncalibrated=payload.get("uncalibrated", {}),
        candidate_cv_pr_auc=payload.get("candidate_cv_pr_auc", {}),
        data_provenance=payload.get("data_provenance", {}),
    )


@router.get(
    "/policy",
    summary="Active credit policy and assumptions",
    description=(
        "The risk grade master scale, the LGD/EAD assumptions and the decision "
        "economics currently in force. Everything here is configuration, not "
        "code, and is versioned in config/risk_policy.yaml."
    ),
    responses=ERRORS,
)
def policy(service: ScoringService = Depends(get_scoring_service)) -> dict:
    return {
        "grade_scale": [
            {"grade": g.grade, "max_pd": g.max_pd, "description": g.description}
            for g in grade_scale(service.policy)
        ],
        "loss_assumptions": assumption_disclosure(service.policy),
        "decision_policy": policy_disclosure(service.policy),
    }
