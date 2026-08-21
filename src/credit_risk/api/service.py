"""Scoring service: the business logic, with no HTTP in it.

Keeping this free of FastAPI types means the whole decisioning path can be
tested without spinning up a client, and the same service backs the single,
batch and stress endpoints.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from credit_risk.config import load_risk_policy
from credit_risk.models.explain import PredictionExplainer
from credit_risk.models.registry import ModelMetadata
from credit_risk.risk.expected_loss import assumption_disclosure, compute_loss_components
from credit_risk.risk.grades import assign_grade, grade_scale
from credit_risk.risk.policy import decide, policy_disclosure

logger = logging.getLogger(__name__)


class ScoringService:
    """Turns validated loan applications into credit decisions."""

    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata,
        metrics: dict[str, Any],
        explainer: PredictionExplainer | None = None,
    ):
        self.model = model
        self.metadata = metadata
        self.metrics = metrics
        self.explainer = explainer
        self.policy = load_risk_policy()
        self._grade_descriptions = {g.grade: g.description for g in grade_scale(self.policy)}
        self._feature_columns = list(metadata.feature_columns)

    # ------------------------------------------------------------------ utils

    def to_frame(self, applications: list[dict[str, Any]]) -> pd.DataFrame:
        """Build a model-ready frame, with columns in the trained order.

        Column order matters: the fitted ColumnTransformer selects by name, but
        an explicit reindex keeps behaviour identical between single and batch
        paths and fails loudly if a field is missing.
        """
        frame = pd.DataFrame(applications)
        missing = [c for c in self._feature_columns if c not in frame.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        return frame[self._feature_columns]

    def predict_pd(self, frame: pd.DataFrame) -> Any:
        """Calibrated probability of default."""
        return self.model.predict_proba(frame)[:, 1]

    # -------------------------------------------------------------- assessment

    def assess_one(
        self,
        application: dict[str, Any],
        *,
        request_id: str | None = None,
        explain: bool = True,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())

        frame = self.to_frame([application])
        pd_value = float(self.predict_pd(frame)[0])

        loss = compute_loss_components(
            pd_value,
            float(frame["loan_amount"].iloc[0]),
            float(frame["property_value"].iloc[0]) if "property_value" in frame else None,
            self.policy,
        )
        decision = decide(loss.pd, loss.lgd, loss.ead, self.policy)
        grade = assign_grade(pd_value, self.policy)

        explanation = None
        if explain and self.explainer is not None and self.explainer.available:
            try:
                explanation = self.explainer.explain(frame, self._feature_columns).to_dict()
            except Exception:
                logger.exception("explanation failed; returning assessment without reason codes")

        latency = (time.perf_counter() - started) * 1000.0
        logger.info(
            "assessed application",
            extra={
                "request_id": request_id,
                "model_version": self.metadata.model_version,
                "pd": round(pd_value, 6),
                "risk_grade": grade,
                "decision": str(decision.decision),
                "expected_loss": round(loss.expected_loss, 2),
                "latency_ms": round(latency, 2),
                "event": "credit_decision",
            },
        )

        return {
            "request_id": request_id,
            "model_version": self.metadata.model_version,
            "assessed_at": datetime.now(UTC).isoformat(),
            "probability_of_default": pd_value,
            "risk_grade": grade,
            "grade_description": self._grade_descriptions.get(grade, ""),
            "loss": loss.to_dict(),
            "decision": {
                "decision": str(decision.decision),
                "reason": decision.reason,
                "break_even_pd": decision.break_even_pd,
                "expected_profit": decision.expected_profit,
                "expected_revenue": decision.expected_revenue,
                "expected_loss": decision.expected_loss,
            },
            "explanation": explanation,
            "assumptions": {
                **assumption_disclosure(self.policy),
                "decision_policy": policy_disclosure(self.policy),
            },
            "latency_ms": round(latency, 3),
        }

    def assess_batch(
        self,
        applications: list[dict[str, Any]],
        *,
        request_id: str | None = None,
        explain: bool = False,
    ) -> dict[str, Any]:
        """Score many applications in one vectorised pass.

        A single malformed row returns a per-row error rather than failing the
        whole batch.
        """
        started = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())

        frame = self.to_frame(applications)
        pd_values = self.predict_pd(frame)

        results: list[dict[str, Any]] = []
        succeeded = 0
        for i, application in enumerate(applications):
            try:
                assessment = self._assemble(
                    application, float(pd_values[i]), frame.iloc[[i]], request_id, explain
                )
                results.append({"index": i, "assessment": assessment, "error": None})
                succeeded += 1
            except Exception as exc:
                logger.warning("batch row %d failed: %s", i, exc)
                results.append({"index": i, "assessment": None, "error": str(exc)})

        portfolio_summary = None
        if succeeded:
            from credit_risk.risk.portfolio import aggregate

            ok = [r["index"] for r in results if r["assessment"] is not None]
            segment_cols = [
                c for c in ("Region", "loan_purpose", "occupancy_type") if c in frame.columns
            ]
            portfolio_summary = aggregate(
                pd_values[ok],
                frame["loan_amount"].iloc[ok],
                frame["property_value"].iloc[ok] if "property_value" in frame else None,
                segments=frame[segment_cols].iloc[ok].reset_index(drop=True)
                if segment_cols
                else None,
                policy=self.policy,
            ).to_dict()

        latency = (time.perf_counter() - started) * 1000.0
        logger.info(
            "assessed batch",
            extra={
                "request_id": request_id,
                "n_submitted": len(applications),
                "n_succeeded": succeeded,
                "latency_ms": round(latency, 2),
                "event": "batch_credit_decision",
            },
        )

        return {
            "request_id": request_id,
            "model_version": self.metadata.model_version,
            "n_submitted": len(applications),
            "n_succeeded": succeeded,
            "n_failed": len(applications) - succeeded,
            "results": results,
            "portfolio": portfolio_summary,
            "latency_ms": round(latency, 3),
        }

    def _assemble(
        self,
        application: dict[str, Any],
        pd_value: float,
        row: pd.DataFrame,
        request_id: str,
        explain: bool,
    ) -> dict[str, Any]:
        loss = compute_loss_components(
            pd_value,
            float(row["loan_amount"].iloc[0]),
            float(row["property_value"].iloc[0]) if "property_value" in row else None,
            self.policy,
        )
        decision = decide(loss.pd, loss.lgd, loss.ead, self.policy)
        grade = assign_grade(pd_value, self.policy)

        explanation = None
        if explain and self.explainer is not None and self.explainer.available:
            explanation = self.explainer.explain(row, self._feature_columns).to_dict()

        return {
            "request_id": request_id,
            "model_version": self.metadata.model_version,
            "assessed_at": datetime.now(UTC).isoformat(),
            "probability_of_default": pd_value,
            "risk_grade": grade,
            "grade_description": self._grade_descriptions.get(grade, ""),
            "loss": loss.to_dict(),
            "decision": {
                "decision": str(decision.decision),
                "reason": decision.reason,
                "break_even_pd": decision.break_even_pd,
                "expected_profit": decision.expected_profit,
                "expected_revenue": decision.expected_revenue,
                "expected_loss": decision.expected_loss,
            },
            "explanation": explanation,
            "assumptions": {
                **assumption_disclosure(self.policy),
                "decision_policy": policy_disclosure(self.policy),
            },
            "latency_ms": None,
        }
