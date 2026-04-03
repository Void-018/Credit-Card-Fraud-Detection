from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI(title="Credit Card Fraud Detection API")

# Load model and scaler
try:
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')
    print("Model and scaler loaded.")
except FileNotFoundError:
    print("Model/scaler not found. Train first with python train_model.py")
    raise

class Transaction(BaseModel):
    features: list[float]  # 30 features

@app.post("/predict")
def predict(transaction: Transaction):
    if len(transaction.features) != 30:
        return {"error": "Exactly 30 features required"}
    
    input_array = np.array([transaction.features])
    scaled = scaler.transform(input_array)
    
    pred = model.predict(scaled)[0]
    fraud = 1 if pred == -1 else 0  # Map IsolationForest -1(fraud)->1, 1(normal)->0
    score = -model.decision_function(scaled)[0]  # Higher = more anomalous/fraud prob-like [0,1] normalized roughly
    
    return {
        "fraud": fraud,
        "anomaly_score": float(score),
        "prediction": "FRAUD" if fraud == 1 else "Non-fraud"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

