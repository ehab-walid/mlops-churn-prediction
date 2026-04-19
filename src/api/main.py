from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Churn Prediction API")

# THE MAGIC LINE: Just load the physical file. No MLflow needed!
print("Loading model...")
model = joblib.load("models/churn_pipeline.joblib")
print("Model loaded successfully!")

class CustomerData(BaseModel):
    gender: str = "Female"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 29.85
    TotalCharges: float = 29.85

@app.post("/predict")
def predict_churn(customer: CustomerData):
    df = pd.DataFrame([customer.dict()])
    prediction = model.predict(df)
    churn_prob = prediction[0]
    
    return {
        "customer_input": customer.dict(),
        "will_churn": bool(churn_prob),
        "message": "High risk of leaving!" if churn_prob == 1 else "Customer is likely to stay."
    }