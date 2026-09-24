from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="UPI Fraud Detection API",
    description="Real-time UPI transaction fraud detection",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Load trained model
model = joblib.load("models/upi_fraud_model.pkl")


class Transaction(BaseModel):
    amount: float
    transaction_hour: int
    merchant_category: str
    device_type: str
    location: str
    failed_attempts: int
    account_age_days: int


@app.get("/")
def home():
    return {
        "message": "UPI Fraud Detection API is running",
        "status": "active"
    }


@app.post("/predict")
def predict(transaction: Transaction):

    data = pd.DataFrame([{
        "amount": transaction.amount,
        "transaction_hour": transaction.transaction_hour,
        "merchant_category": transaction.merchant_category,
        "device_type": transaction.device_type,
        "location": transaction.location,
        "failed_attempts": transaction.failed_attempts,
        "account_age_days": transaction.account_age_days
    }])

    probability = float(model.predict_proba(data)[0][1])

    if probability >= 0.75:
        risk_level = "HIGH"
        action = "BLOCK"
    elif probability >= 0.55:
        risk_level = "MEDIUM"
        action = "REVIEW"
    else:
        risk_level = "LOW"
        action = "APPROVE"

    return {
        "fraud_probability": round(probability, 4),
        "risk_level": risk_level,
        "action": action
    }