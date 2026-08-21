"""Dependency injection for the API.

The original app kept the model in a module-level global mutated by the lifespan
handler, which made it impossible to substitute in tests and meant a failed load
left the service running with ``model = None``.

Here state lives on ``app.state`` and is reached through FastAPI dependencies,
so tests can override the service and a failed load makes readiness fail rather
than silently degrading every prediction to a 500.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import Depends, HTTPException, Request, status

from credit_risk.api.service import ScoringService
from credit_risk.config import Settings
from credit_risk.models.explain import PredictionExplainer
from credit_risk.models.registry import ModelRegistry

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    """Everything loaded at startup, held on ``app.state``."""

    service: ScoringService | None = None
    model_version: str | None = None
    load_error: str | None = None
    canary_ok: bool = False

    @property
    def ready(self) -> bool:
        return self.service is not None and self.canary_ok


def _canary_record(service: ScoringService) -> dict[str, Any]:
    """A known-good application used to prove the model can actually score.

    Built from the data contract so it stays valid as the contract changes.
    """
    from credit_risk.api.schemas.requests import EXAMPLE_VALUES

    return {c: EXAMPLE_VALUES[c] for c in service.metadata.feature_columns if c in EXAMPLE_VALUES}


def load_model_state(settings: Settings) -> ModelState:
    """Load the model artifact and verify it scores. Never raises."""
    state = ModelState()
    try:
        registry = ModelRegistry(settings.artifacts_dir)
        model, metadata, metrics = registry.load(settings.model_version)

        explainer: PredictionExplainer | None
        try:
            explainer = PredictionExplainer(model)
        except Exception:
            logger.exception("explainer unavailable; assessments will omit reason codes")
            explainer = None

        service = ScoringService(model, metadata, metrics, explainer)
        state.service = service
        state.model_version = metadata.model_version

        # Readiness means "can score", not "file exists".
        try:
            record = _canary_record(service)
            result = service.assess_one(record, explain=False)
            state.canary_ok = 0.0 <= result["probability_of_default"] <= 1.0
            logger.info(
                "canary scored successfully",
                extra={
                    "pd": result["probability_of_default"],
                    "model_version": metadata.model_version,
                },
            )
        except Exception as exc:
            state.canary_ok = False
            state.load_error = f"canary prediction failed: {exc}"
            logger.exception("canary prediction failed")

    except FileNotFoundError as exc:
        state.load_error = str(exc)
        logger.error("no model artifact available: %s", exc)
    except Exception as exc:
        state.load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("model load failed")

    return state


# --------------------------------------------------------------- dependencies


def get_model_state(request: Request) -> ModelState:
    return request.app.state.model_state


def get_scoring_service(
    state: ModelState = Depends(get_model_state),
) -> ScoringService:
    """The scoring service, or 503 if the model is not usable.

    503 Service Unavailable is the correct code here - the request is fine, the
    server cannot serve it yet. The original code returned 500.
    """
    if state.service is None or not state.ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=state.load_error or "Model is not loaded or failed its canary check.",
        )
    return state.service


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
