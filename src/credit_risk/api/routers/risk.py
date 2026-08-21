"""Credit risk assessment endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from credit_risk.api.dependencies import get_request_id, get_scoring_service
from credit_risk.api.schemas.requests import BatchAssessmentRequest, LoanApplication
from credit_risk.api.schemas.responses import (
    BatchAssessmentResponse,
    ErrorResponse,
    RiskAssessmentResponse,
)
from credit_risk.api.service import ScoringService
from credit_risk.config import Settings, get_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/risk", tags=["risk"])

COMMON_ERRORS = {
    422: {"model": ErrorResponse, "description": "Request failed schema validation."},
    503: {"model": ErrorResponse, "description": "Model not loaded or not ready."},
}


@router.post(
    "/assess",
    response_model=RiskAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess a single loan application",
    description=(
        "Returns a calibrated probability of default, a risk grade, the expected "
        "loss decomposition (PD x LGD x EAD), an approve/refer/decline decision "
        "derived from credit economics, and SHAP reason codes.\n\n"
        "The LGD and EAD figures rest on stated assumptions, which are echoed in "
        "the `assumptions` block of every response."
    ),
    responses=COMMON_ERRORS,
)
def assess(
    application: LoanApplication,  # type: ignore[valid-type]
    explain: bool = True,
    service: ScoringService = Depends(get_scoring_service),
    request_id: str = Depends(get_request_id),
) -> RiskAssessmentResponse:
    payload = application.model_dump(by_alias=True)
    result = service.assess_one(payload, request_id=request_id, explain=explain)
    return RiskAssessmentResponse(**result)


@router.post(
    "/batch",
    response_model=BatchAssessmentResponse,
    status_code=status.HTTP_200_OK,
    summary="Assess a batch of loan applications",
    description=(
        "Scores many applications in a single vectorised pass and returns "
        "portfolio-level aggregates alongside the individual assessments.\n\n"
        "A row that fails scoring returns a per-row error rather than failing the "
        "whole batch. Explanations are off by default because SHAP dominates "
        "latency at volume."
    ),
    responses={
        **COMMON_ERRORS,
        413: {"model": ErrorResponse, "description": "Batch exceeds the configured maximum."},
    },
)
def assess_batch(
    request: BatchAssessmentRequest,
    service: ScoringService = Depends(get_scoring_service),
    settings: Settings = Depends(get_settings),
    request_id: str = Depends(get_request_id),
) -> BatchAssessmentResponse:
    if len(request.applications) > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Batch size {len(request.applications)} exceeds the maximum of "
                f"{settings.max_batch_size}."
            ),
        )

    payloads = [a.model_dump(by_alias=True) for a in request.applications]
    result = service.assess_batch(
        payloads, request_id=request_id, explain=request.include_explanations
    )
    return BatchAssessmentResponse(**result)
