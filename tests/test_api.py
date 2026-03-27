"""Basic tests for the Loan Default Prediction API."""
from fastapi.testclient import TestClient
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {
    "ID": 1,
    "year": 2019,
    "loan_limit": "cf",
    "Gender": "Male",
    "approv_in_adv": "pre",
    "loan_type": "type1",
    "loan_purpose": "p1",
    "Credit_Worthiness": "l1",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "loan_amount": 150000.0,
    "rate_of_interest": 3.5,
    "Interest_rate_spread": 0.5,
    "Upfront_charges": 1200.0,
    "term": 360.0,
    "Neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "property_value": 250000.0,
    "construction_type": "sb",
    "occupancy_type": "pr",
    "Secured_by": "home",
    "total_units": "1U",
    "income": 80000.0,
    "credit_type": "EXP",
    "co-applicant_credit_type": "EXP",
    "age": "35-44",
    "submission_of_application": "to_inst",
    "LTV": 80.0,
    "Region": "North",
    "Security_Type": "direct",
    "dtir1": 35.0
}

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_predict():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
