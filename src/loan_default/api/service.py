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

import numpy as np
import pandas as pd

from loan_default.config import load_risk_policy
from loan_default.models.explain import PredictionExplainer
from loan_default.models.registry import ModelMetadata
from loan_default.risk.expected_loss import assumption_disclosure, compute_loss_components
from loan_default.risk.grades import assign_grade, grade_scale
from loan_default.risk.policy import decide, policy_disclosure

logger = logging.getLogger(__name__)


class ScoringService:
    """Turns validated loan applications into credit decisions."""

    def __init__(
        self,
        model: Any,
        metadata: ModelMetadata,
        metrics: dict[str, Any],
        explainer: PredictionExplainer | None = None,
        baseline: pd.DataFrame | None = None,
    ):
        self.model = model
        self.metadata = metadata
        self.metrics = metrics
        self.explainer = explainer
        # Training-distribution reference for drift checks. Optional: an older
        # artifact predates baseline capture and must still serve predictions.
        self.baseline = baseline
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
            "decision": decision.to_dict(),
            "explanation": explanation,
            "assumptions_version": str(self.policy.get("version", "unversioned")),
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
        positions: list[int] | None = None,
        validation_errors: dict[int, str] | None = None,
        total_submitted: int | None = None,
    ) -> dict[str, Any]:
        """Score a batch in one vectorised pass.

        ``applications`` holds only the rows that passed validation.
        ``positions`` maps each of them back to its index in the original
        request, and ``validation_errors`` carries the rows that did not, so the
        response can report failures in place without discarding good rows.
        """
        started = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        validation_errors = validation_errors or {}
        positions = positions if positions is not None else list(range(len(applications)))
        total = total_submitted if total_submitted is not None else len(applications)

        results: list[dict[str, Any]] = [
            {"index": index, "assessment": None, "error": message}
            for index, message in validation_errors.items()
        ]
        scored_rows: list[int] = []
        pd_values = None

        if applications:
            frame = self.to_frame(applications)
            pd_values = self.predict_pd(frame)

            for offset, application in enumerate(applications):
                index = positions[offset]
                try:
                    assessment = self._assemble(
                        application,
                        float(pd_values[offset]),
                        frame.iloc[[offset]],
                        request_id,
                        explain,
                    )
                    results.append({"index": index, "assessment": assessment, "error": None})
                    scored_rows.append(offset)
                except Exception as exc:
                    logger.warning("batch row %d failed during scoring: %s", index, exc)
                    results.append({"index": index, "assessment": None, "error": str(exc)})

        results.sort(key=lambda row: row["index"])
        succeeded = len(scored_rows)

        portfolio_summary = None
        if succeeded and pd_values is not None:
            from loan_default.risk.portfolio import aggregate

            segment_cols = [
                c for c in ("Region", "loan_purpose", "occupancy_type") if c in frame.columns
            ]
            rows = np.asarray(scored_rows, dtype=int)
            portfolio_summary = aggregate(
                pd_values[rows],
                frame["loan_amount"].iloc[rows],
                frame["property_value"].iloc[rows] if "property_value" in frame else None,
                segments=frame[segment_cols].iloc[rows].reset_index(drop=True)
                if segment_cols
                else None,
                policy=self.policy,
            ).to_dict()

        latency = (time.perf_counter() - started) * 1000.0
        logger.info(
            "assessed batch",
            extra={
                "request_id": request_id,
                "n_submitted": total,
                "n_succeeded": succeeded,
                "n_rejected": len(validation_errors),
                "latency_ms": round(latency, 2),
                "event": "batch_credit_decision",
            },
        )

        return {
            "request_id": request_id,
            "model_version": self.metadata.model_version,
            "n_submitted": total,
            "n_succeeded": succeeded,
            "n_failed": total - succeeded,
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
            "decision": decision.to_dict(),
            "explanation": explanation,
            "assumptions_version": str(self.policy.get("version", "unversioned")),
            "assumptions": {
                **assumption_disclosure(self.policy),
                "decision_policy": policy_disclosure(self.policy),
            },
            "latency_ms": None,
        }
