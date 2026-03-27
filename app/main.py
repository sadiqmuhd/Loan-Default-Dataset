from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import os
from schema import LoanRequest, PredictionResponse
from feature_engineering import add_features  # import your feature function

app = FastAPI(
    title="Loan Default Prediction API",
    description="API for predicting loan default using XGBoost",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
try:
    with open('model.pkl', 'rb') as f:
        model_pipeline = pickle.load(f)
    print("✓ Model loaded successfully!")
except Exception as e:
    model_pipeline = None
    print(f"ERROR loading model: {e}")

@app.get("/")
def read_root():
    return {"message": "Loan Default Prediction API", "model_loaded": model_pipeline is not None}

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model_pipeline is not None else "unhealthy",
        "model_loaded": model_pipeline is not None,
        "files_exist": {
            "model.pkl": os.path.exists('model.pkl'),
            "preprocessor.pkl": os.path.exists('preprocessor.pkl')
        }
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(loan_data: LoanRequest):
    if model_pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded!")

    try:
        # Convert request to dataframe
        input_df = pd.DataFrame([loan_data.model_dump(by_alias=True)])

        # Feature engineering
        input_fe = add_features(input_df)

        # Predict
        pred = model_pipeline.predict(input_fe)[0]
        prob = model_pipeline.predict_proba(input_fe)[0][1]

        # Risk level
        if prob < 0.3:
            risk_level = "Low Risk"
        elif prob < 0.6:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"

        return PredictionResponse(prediction=int(pred), probability=float(prob), risk_level=risk_level)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
