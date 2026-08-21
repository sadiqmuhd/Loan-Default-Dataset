"""Liveness and readiness probes.

These are deliberately separate. The original single ``/health`` endpoint
conflated the two and checked ``os.path.exists`` on the artifact files - which
reported healthy for a ``preprocessor.pkl`` that could not actually be
unpickled, and for a container that shipped without the model at all.

  /health/live   process is up and serving. Never touches the model.
  /health/ready  the model is loaded AND scored a canary record successfully.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from credit_risk.api.dependencies import ModelState, get_model_state
from credit_risk.api.schemas.responses import HealthResponse, ReadinessResponse
from credit_risk.config import Settings, get_settings

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
