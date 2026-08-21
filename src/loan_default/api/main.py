"""FastAPI application factory.

This module is deliberately thin: wiring only. Business logic lives in
``service.py``, risk methodology in ``loan_default.risk``, and HTTP concerns in
the routers, middleware and error handlers.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from loan_default.api.dependencies import load_model_state
from loan_default.api.errors import register_exception_handlers
from loan_default.api.middleware import RequestContextMiddleware
from loan_default.api.routers import health, model, portfolio, risk
from loan_default.config import Settings, get_settings
from loan_default.logging_config import configure_logging

logger = logging.getLogger(__name__)

DESCRIPTION = """
Credit risk decisioning for mortgage applications.

Given a loan application, the service returns a **calibrated probability of
default**, a **risk grade**, the **expected loss** decomposition
(EL = PD x LGD x EAD), an **approve / refer / decline** decision derived from
credit economics, and **SHAP reason codes** supporting that decision.

### What is modelled and what is assumed

Only **PD** is modelled. **LGD** is a collateral-based proxy and **EAD** is the
origination amount, because the dataset contains no recovery cash flows, workout
costs or balance history. Every assumption is returned in the `assumptions`
block of each response and is configured in `config/risk_policy.yaml`.

### Fair lending

Gender and age are **not collected by this API** and are excluded from the model
as a fair-lending safeguard. Measured cost of exclusion: 4.6 basis points of
ROC-AUC. This is a design decision, not a claim of regulatory compliance.

### Known limitations

* `Credit_Score` is not predictive in the training data (univariate ROC-AUC
  0.503) and appears to be randomly generated.
* The dataset has no time dimension, so no out-of-time validation or
  macroeconomic conditioning is possible.

See `MODEL_CARD.md` for the full model card.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup and verify it can actually score."""
    settings: Settings = get_settings()
    configure_logging(level=settings.log_level, json_logs=settings.json_logs)

    logger.info("starting %s v%s", settings.api_title, settings.api_version)
    app.state.model_state = load_model_state(settings)

    if app.state.model_state.ready:
        logger.info("model ready", extra={"model_version": app.state.model_state.model_version})
    else:
        # Start anyway so /health/ready can report *why* - but readiness fails,
        # so an orchestrator will not send traffic here.
        logger.error("starting WITHOUT a usable model: %s", app.state.model_state.load_error)

    yield

    logger.info("shutting down")
    app.state.model_state = None


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title=settings.api_title,
        description=DESCRIPTION,
        version=settings.api_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        contact={"name": "Abubakar Sadiq Muhammad"},
        license_info={"name": "MIT"},
    )

    # Explicit origins, and credentials disabled. The original app combined
    # allow_origins=["*"] with allow_credentials=True, which browsers reject.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)
    register_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(risk.router)
    app.include_router(model.router)
    app.include_router(portfolio.router)

    @app.get("/", tags=["meta"], summary="Service metadata")
    def root() -> dict:
        return {
            "service": settings.api_title,
            "version": settings.api_version,
            "docs": "/docs",
            "endpoints": [
                "POST /v1/risk/assess",
                "POST /v1/risk/batch",
                "GET  /v1/model/metadata",
                "GET  /v1/model/metrics",
                "GET  /v1/model/policy",
                "POST /v1/portfolio/stress-test",
                "GET  /v1/portfolio/summary",
                "GET  /health/live",
                "GET  /health/ready",
            ],
        }

    return app


app = create_app()
