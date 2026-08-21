"""API contract tests."""

from __future__ import annotations

import pytest

from tests.conftest import requires_model

pytestmark = requires_model


# ------------------------------------------------------------------- health


def test_liveness_is_always_ok(app_client):
    response = app_client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_reports_a_working_model(app_client):
    """Readiness must verify the model SCORES, not that a file exists."""
    response = app_client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["model_loaded"] is True
    assert body["canary_prediction_ok"] is True
    assert body["model_version"]


def test_root_lists_endpoints(app_client):
    body = app_client.get("/").json()
    assert "POST /v1/risk/assess" in body["endpoints"]


# ------------------------------------------------------------------- assess


def test_assess_returns_full_assessment(app_client, example_application):
    response = app_client.post("/v1/risk/assess", json=example_application)
    assert response.status_code == 200
    body = response.json()

    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert body["risk_grade"] in list("ABCDEFG")
    assert body["decision"]["decision"] in {"APPROVE", "REFER", "DECLINE"}
    assert body["model_version"]
    assert body["request_id"]
    assert body["assessed_at"]


def test_expected_loss_identity_holds_in_the_response(app_client, example_application):
    """EL must equal PD x LGD x EAD in the payload the caller actually receives."""
    body = app_client.post("/v1/risk/assess", json=example_application).json()
    loss = body["loss"]
    assert loss["expected_loss"] == pytest.approx(loss["pd"] * loss["lgd"] * loss["ead"], rel=1e-6)


def test_assumptions_are_disclosed(app_client, example_application):
    """Any response quoting a loss figure must disclose what it rests on."""
    body = app_client.post("/v1/risk/assess", json=example_application).json()
    assumptions = body["assumptions"]
    assert assumptions["lgd_is_modelled"] is False
    assert "proxy" in assumptions["lgd_note"].lower()
    assert "decision_policy" in assumptions


def test_explanation_contains_reason_codes(app_client, example_application):
    body = app_client.post("/v1/risk/assess", json=example_application).json()
    explanation = body["explanation"]
    assert explanation is not None
    assert explanation["risk_drivers"] or explanation["risk_reducers"]
    for reason in explanation["risk_drivers"]:
        assert reason["direction"] == "increases_risk"
        assert reason["label"]
    for reason in explanation["risk_reducers"]:
        assert reason["direction"] == "reduces_risk"


def test_explanations_can_be_disabled(app_client, example_application):
    response = app_client.post(
        "/v1/risk/assess", json=example_application, params={"explain": False}
    )
    assert response.json()["explanation"] is None


def test_request_id_is_echoed(app_client, example_application):
    response = app_client.post(
        "/v1/risk/assess", json=example_application, headers={"X-Request-ID": "trace-abc-123"}
    )
    assert response.headers["X-Request-ID"] == "trace-abc-123"
    assert response.json()["request_id"] == "trace-abc-123"


def test_response_time_header_present(app_client, example_application):
    response = app_client.post("/v1/risk/assess", json=example_application)
    assert float(response.headers["X-Response-Time-ms"]) > 0


def test_identical_requests_are_deterministic(app_client, example_application):
    first = app_client.post("/v1/risk/assess", json=example_application).json()
    second = app_client.post("/v1/risk/assess", json=example_application).json()
    assert first["probability_of_default"] == second["probability_of_default"]
    assert first["risk_grade"] == second["risk_grade"]


# --------------------------------------------------------------- validation


def test_invalid_category_returns_422_with_field_detail(app_client, example_application):
    response = app_client.post(
        "/v1/risk/assess", json={**example_application, "credit_type": "STD"}
    )
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "validation_error"
    assert body["request_id"]
    assert any("credit_type" in (d.get("field") or "") for d in body["details"])


def test_negative_loan_amount_rejected(app_client, example_application):
    response = app_client.post(
        "/v1/risk/assess", json={**example_application, "loan_amount": -1000}
    )
    assert response.status_code == 422


def test_missing_required_field_rejected(app_client, example_application):
    payload = {k: v for k, v in example_application.items() if k != "loan_amount"}
    assert app_client.post("/v1/risk/assess", json=payload).status_code == 422


def test_protected_attribute_is_rejected(app_client, example_application):
    """ECOA: the service must refuse to receive gender at all."""
    response = app_client.post("/v1/risk/assess", json={**example_application, "Gender": "Male"})
    assert response.status_code == 422


def test_error_response_never_leaks_internals(app_client, example_application):
    body = app_client.post("/v1/risk/assess", json={**example_application, "LTV": 9999}).json()
    serialised = str(body).lower()
    for leak in ("traceback", "site-packages", "c:\\", "/usr/lib"):
        assert leak not in serialised


# -------------------------------------------------------------------- batch


def test_batch_scores_multiple_applications(app_client, example_application):
    response = app_client.post("/v1/risk/batch", json={"applications": [example_application] * 5})
    assert response.status_code == 200
    body = response.json()
    assert body["n_submitted"] == 5
    assert body["n_succeeded"] == 5
    assert body["n_failed"] == 0
    assert len(body["results"]) == 5


def test_batch_returns_portfolio_aggregates(app_client, example_application):
    body = app_client.post(
        "/v1/risk/batch", json={"applications": [example_application] * 10}
    ).json()
    portfolio = body["portfolio"]
    assert portfolio["n_exposures"] == 10
    assert portfolio["total_exposure"] > 0
    assert portfolio["total_expected_loss"] > 0


def test_batch_matches_single_assessment(app_client, example_application):
    """Batch and single paths must agree exactly."""
    single = app_client.post("/v1/risk/assess", json=example_application).json()
    batch = app_client.post("/v1/risk/batch", json={"applications": [example_application]}).json()
    assert batch["results"][0]["assessment"]["probability_of_default"] == pytest.approx(
        single["probability_of_default"]
    )


def test_empty_batch_rejected(app_client):
    assert app_client.post("/v1/risk/batch", json={"applications": []}).status_code == 422


def test_oversized_batch_rejected(app_client, example_application, monkeypatch):
    from credit_risk.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings, "max_batch_size", 3)
    response = app_client.post("/v1/risk/batch", json={"applications": [example_application] * 4})
    assert response.status_code == 413


# ------------------------------------------------------------- governance


def test_model_metadata_exposes_provenance(app_client):
    body = app_client.get("/v1/model/metadata").json()
    assert len(body["data_sha256"]) == 64
    assert body["n_training_rows"] > 0
    assert body["calibration_method"]
    assert body["assumptions"]
    assert body["limitations"]


def test_metadata_documents_exclusions(app_client):
    excluded = app_client.get("/v1/model/metadata").json()["excluded_columns"]
    assert "Gender" in excluded["protected"]
    assert "Interest_rate_spread" in excluded["leakage"]


def test_model_metrics_are_published(app_client):
    body = app_client.get("/v1/model/metrics").json()
    calibrated = body["calibrated"]
    assert 0.70 < calibrated["roc_auc"] < 0.95
    assert calibrated["brier_score"] > 0
    assert body["candidate_cv_pr_auc"]


def test_policy_endpoint_publishes_the_grade_scale(app_client):
    body = app_client.get("/v1/model/policy").json()
    assert len(body["grade_scale"]) == 7
    assert body["loss_assumptions"]["lgd_is_modelled"] is False


def test_openapi_schema_is_valid(app_client):
    schema = app_client.get("/openapi.json").json()
    assert "/v1/risk/assess" in schema["paths"]
    assert "/v1/portfolio/stress-test" in schema["paths"]
