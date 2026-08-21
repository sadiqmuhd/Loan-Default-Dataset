"""Drift monitoring against the model's own training distribution.

A drift number is only meaningful relative to a specific reference. The baseline
here is a sample of the rows the deployed model was actually fitted on, saved
into the artifact at training time, so a model and its reference distribution
can never be mismatched.

WHAT THIS DOES NOT DO: there is no scheduled job, no metric store and no
alerting. The endpoint scores a batch you submit. Production monitoring would
run this on a window of live traffic and page someone on a threshold breach;
that is infrastructure this project does not have, and claiming otherwise would
be dishonest.
"""

from __future__ import annotations

import logging

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, status

from loan_default.api.dependencies import get_request_id, get_scoring_service
from loan_default.api.schemas.requests import BatchAssessmentRequest, validate_application
from loan_default.api.service import ScoringService
from loan_default.monitoring.drift import feature_drift, prediction_drift

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/monitoring", tags=["monitoring"])

# PSI convention, widely used in credit scorecard monitoring.
PSI_BANDS = {
    "none": "< 0.10 - no material shift",
    "moderate": "0.10 to 0.25 - investigate",
    "major": "> 0.25 - population has shifted materially",
}


@router.post(
    "/drift",
    summary="Population stability of a batch against the training distribution",
)
def assess_drift(
    request: BatchAssessmentRequest,
    service: ScoringService = Depends(get_scoring_service),
    request_id: str = Depends(get_request_id),
) -> dict:
    """Compare a batch of applications to the model's training baseline.

    Returns per-feature PSI worst-first, plus PSI on the predicted PD
    distribution, which catches shifts that no single feature reveals.
    """
    if service.baseline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "This model artifact carries no drift baseline. Retrain to "
                "capture one: python -m loan_default.models.train"
            ),
        )

    # Invalid rows are skipped rather than reported: this endpoint answers
    # "has the population shifted", and a malformed row is a data quality
    # problem for /v1/risk/batch to surface, not a distributional signal.
    rows: list[dict] = []
    for payload in request.applications:
        cleaned, _error = validate_application(payload)
        if cleaned is not None:
            rows.append(cleaned)

    if not rows:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid applications in the batch; nothing to compare.",
        )

    current = pd.DataFrame(rows)
    baseline = service.baseline
    reference_pd = baseline["_reference_pd"]
    reference_features = baseline.drop(columns=["_reference_pd"])

    shared = [c for c in reference_features.columns if c in current.columns]
    drift = feature_drift(reference_features, current, columns=shared)

    current_pd = service.predict_pd(service.to_frame(rows))
    pd_drift = prediction_drift(reference_pd, current_pd)

    worst = drift[0] if drift else None
    logger.info(
        "drift assessed",
        extra={
            "request_id": request_id,
            "n_rows": len(current),
            "worst_feature": worst.feature if worst else None,
            "worst_psi": round(worst.psi, 4) if worst else None,
            "prediction_psi": round(pd_drift.psi, 4),
            "event": "drift_check",
        },
    )

    return {
        "request_id": request_id,
        "model_version": service.metadata.model_version,
        "n_rows_compared": len(current),
        "n_baseline_rows": len(baseline),
        "prediction_drift": {
            "psi": pd_drift.psi,
            "severity": pd_drift.severity,
            "reference_mean_pd": float(reference_pd.mean()),
            "current_mean_pd": float(current_pd.mean()),
        },
        "feature_drift": [
            {
                "feature": d.feature,
                "psi": d.psi,
                "severity": d.severity,
                "kind": d.kind,
            }
            for d in drift
        ],
        "psi_bands": PSI_BANDS,
        "note": (
            "Baseline is a sample of the rows this model was fitted on. PSI "
            "measures input shift only - it says nothing about whether accuracy "
            "has degraded, which needs realised outcomes."
        ),
    }
