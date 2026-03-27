from pydantic import BaseModel, Field
from typing import Optional, Literal

class LoanRequest(BaseModel):
    ID: int
    year: int
    loan_limit: Literal['cf', 'nf']
    Gender: Literal['Male', 'Female']
    approv_in_adv: Literal['nopre', 'pre']
    loan_type: Literal['type1', 'type2', 'type3']
    loan_purpose: Literal['p1', 'p2', 'p3']
    Credit_Worthiness: Literal['l1', 'l2', 'l3']
    open_credit: Literal['nopc', 'opc']
    business_or_commercial: Literal['nob/c', 'b/c']
    loan_amount: float
    rate_of_interest: Optional[float] = None
    Interest_rate_spread: Optional[float] = None
    Upfront_charges: Optional[float] = None
    term: float
    Neg_ammortization: Literal['not_neg', 'neg']
    interest_only: Literal['not_int', 'int_only']
    lump_sum_payment: Literal['not_lpsm', 'lpsm']
    property_value: Optional[float] = None
    construction_type: Literal['sb', 'other']
    occupancy_type: Literal['pr', 'sr']
    Secured_by: Literal['home', 'other']
    total_units: Literal['1U', '2U', '3U']
    income: float
    credit_type: Literal['EXP', 'STD', 'OTH']
    Credit_Score: int
    co_applicant_credit_type: Literal['CIB', 'NONE'] = Field(alias="co-applicant_credit_type")
    age: Literal['25-34', '35-44', '45-54']
    submission_of_application: Literal['to_inst', 'to_bank']
    LTV: Optional[float] = None
    Region: Literal['north', 'south', 'east', 'west']
    Security_Type: Literal['direct', 'indirect']
    dtir1: Optional[float] = None

    class Config:
        populate_by_name = True
        schema_extra = {
            "example": {
                "ID": 1,
                "year": 2019,
                "loan_limit": "cf",
                "Gender": "Male",
                "approv_in_adv": "nopre",
                "loan_type": "type1",
                "loan_purpose": "p1",
                "Credit_Worthiness": "l1",
                "open_credit": "nopc",
                "business_or_commercial": "nob/c",
                "loan_amount": 150000.0,
                "rate_of_interest": 3.5,
                "Interest_rate_spread": 0.5,
                "Upfront_charges": 1500.0,
                "term": 360.0,
                "Neg_ammortization": "not_neg",
                "interest_only": "not_int",
                "lump_sum_payment": "not_lpsm",
                "property_value": 200000.0,
                "construction_type": "sb",
                "occupancy_type": "pr",
                "Secured_by": "home",
                "total_units": "1U",
                "income": 80000.0,
                "credit_type": "EXP",
                "Credit_Score": 700,
                "co-applicant_credit_type": "CIB",
                "age": "25-34",
                "submission_of_application": "to_inst",
                "LTV": 75.0,
                "Region": "south",
                "Security_Type": "direct",
                "dtir1": 35.0
            }
        }

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
