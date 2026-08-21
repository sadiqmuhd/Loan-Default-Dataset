"""Portfolio-level analytics: stress testing and concentration.

These endpoints score a sample of the held-out portfolio rather than accepting
one from the caller, because the point is to exercise the model against a
realistic book. In a production deployment this would read from a portfolio
store instead of the training file.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from loan_default.api.dependencies import get_scoring_service
from loan_default.api.schemas.requests import StressTestRequest
from loan_default.api.schemas.responses import ErrorResponse
from loan_default.api.service import ScoringService
from loan_default.config import get_settings, load_stress_scenarios
from loan_default.data.loader import load_dataset
from loan_default.risk import stress
from loan_default.risk.portfolio import aggregate

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/portfolio", tags=["portfolio"])

SEGMENTS = ("Region", "loan_purpose", "occupancy_type")


def _sample_portfolio(service: ScoringService, n: int):
    """Sample a portfolio down to the model's feature columns.

    In a production deployment this would query a portfolio store. Here it reads
    the full dataset when present, and otherwise falls back to the committed
    5,000-row stratified sample - so a deployed instance built straight from the
    repository still has a working portfolio, without carrying 28MB of CSV.
    """
    settings = get_settings()
    for path in (settings.data_path, settings.portfolio_sample_path):
        try:
            dataset = load_dataset(path)
        except FileNotFoundError:
            continue
        frame = dataset.X[service.metadata.feature_columns]
        if len(frame) > n:
            frame = frame.sample(n=n, random_state=42)
        return frame.reset_index(drop=True)

    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=(
            "No portfolio data is available to this instance. Expected either "
            f"{settings.data_path.name} or {settings.portfolio_sample_path.name} "
            "under data/."
        ),
    )


@router.post(
    "/stress-test",
    summary="Run portfolio stress scenarios",
    description=(
        "Applies the configured input shocks to a sample of the portfolio, "
        "re-scores every exposure through the PD model, and reports the change "
        "in expected loss, weighted-average PD and grade distribution.\n\n"
        "**This is assumption-driven sensitivity analysis, not a "
        "macro-conditioned forecast, and it is not CCAR or DFAST.** The dataset "
        "has no macroeconomic variables and no time dimension, so PD cannot be "
        "conditioned on a macro path. The limitations are returned with every "
        "result."
    ),
    responses={
        422: {"model": ErrorResponse, "description": "Unknown scenario requested."},
        503: {"model": ErrorResponse, "description": "Model not loaded."},
    },
)
def stress_test(
    request: StressTestRequest,
    service: ScoringService = Depends(get_scoring_service),
) -> dict:
    config = load_stress_scenarios()
    scenarios = config["scenarios"]

    if request.scenarios:
        available = {s["name"] for s in scenarios}
        unknown = set(request.scenarios) - available
        if unknown:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail=f"Unknown scenarios: {sorted(unknown)}. Available: {sorted(available)}",
            )
        # Always keep the base case - the deltas are measured against it.
        selected = [s for s in scenarios if s["name"] in {*request.scenarios, "base"}]
        config = {**config, "scenarios": selected}

    frame = _sample_portfolio(service, request.sample_size)
    segments = frame[[c for c in SEGMENTS if c in frame.columns]]
    results = stress.run_all_scenarios(
        service.model, frame, config, service.policy, segments=segments
    )

    payload: dict = {
        "model_version": service.metadata.model_version,
        "sample_size": len(frame),
        "scenarios": [r.to_dict() for r in results],
        "limitations": stress.LIMITATIONS,
    }

    if request.include_sensitivity:
        sweeps = {}
        for variable, magnitudes in config.get("sensitivity", {}).items():
            sweeps[variable] = stress.sensitivity_sweep(
                service.model, frame, variable, magnitudes, service.policy
            ).to_dict(orient="records")
        payload["sensitivity"] = sweeps

    return payload


@router.get(
    "/summary",
    summary="Portfolio risk summary",
    description=(
        "Scores a sample of the portfolio and returns exposure, expected loss, "
        "the grade distribution and concentration measures (Herfindahl-Hirschman "
        "Index) by region, loan purpose and occupancy type."
    ),
    responses={503: {"model": ErrorResponse, "description": "Model not loaded."}},
)
def portfolio_summary(
    sample_size: int = 5000,
    service: ScoringService = Depends(get_scoring_service),
) -> dict:
    sample_size = max(100, min(sample_size, 50_000))
    frame = _sample_portfolio(service, sample_size)
    pd_values = service.predict_pd(frame)
    segments = frame[[c for c in SEGMENTS if c in frame.columns]]

    summary = aggregate(
        pd_values,
        frame["loan_amount"],
        frame.get("property_value"),
        segments=segments,
        policy=service.policy,
    )
    return {"model_version": service.metadata.model_version, **summary.to_dict()}
