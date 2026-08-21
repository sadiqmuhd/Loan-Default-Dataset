"""Liveness and readiness probes.

  /health/live   the process is up and serving. Never touches the model.
  /health/ready  the model is loaded and has scored a canary record.

Keeping them separate matters: a liveness failure should restart the process, a
readiness failure should only stop traffic being routed to it.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from loan_default.api.dependencies import ModelState, get_model_state
from loan_default.api.schemas.responses import HealthResponse, ReadinessResponse
from loan_default.config import Settings, get_settings

router = APIRouter(tags=["health"])


@router.get(
    "/health/live",
    response_model=HealthResponse,
    summary="Liveness probe",
    description="Returns 200 whenever the process is serving. Does not check the model.",
)
def liveness(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", service=settings.api_title, api_version=settings.api_version)


@router.get(
    "/health/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Returns 200 only when the model artifact is loaded and has successfully "
        "scored a canary record. Returns 503 otherwise, so an orchestrator will "
        "not route traffic to an instance that cannot make predictions."
    ),
    responses={503: {"description": "Model not loaded or canary check failed."}},
)
def readiness(
    response: Response,
    state: ModelState = Depends(get_model_state),
) -> ReadinessResponse:
    if not state.ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return ReadinessResponse(
            status="not_ready",
            model_loaded=state.service is not None,
            model_version=state.model_version,
            canary_prediction_ok=state.canary_ok,
            detail=state.load_error or "Model is not ready to serve.",
        )
    return ReadinessResponse(
        status="ready",
        model_loaded=True,
        model_version=state.model_version,
        canary_prediction_ok=True,
    )
