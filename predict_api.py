from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from stable_baselines3 import DQN
from environment.custom_env import FraudDetectionEnv

# -------------------------------
# 1. Load DQN model and environment
# -------------------------------
model = DQN.load("models/dqn_fraud_model")
env = FraudDetectionEnv()

app = FastAPI(title="Fraud Detection RL API")

# -------------------------------
# 2. Define input transaction schema
# -------------------------------
class Transaction(BaseModel):
    amount: float
    hour: int
    location: int
    risk_score: float

# -------------------------------
# 3. Prediction endpoint
# -------------------------------
@app.post("/predict")
def predict(transaction: Transaction):
    # Prepare transaction as observation
    obs = np.array([transaction.amount,
                    transaction.hour,
                    transaction.location,
                    transaction.risk_score], dtype=np.float32)

    # Set as current transaction in env for consistency
    env.current_transaction = np.append(obs, 0)  # dummy label, won't be used
    action, _ = model.predict(obs)

    label_str = "Fraud" if action == 1 else "Legit"

    return {
        "transaction": transaction.dict(),
        "prediction": label_str
    }

# -------------------------------
# 4. Root endpoint
# -------------------------------
@app.get("/")
def root():
    return {"message": "Welcome to the DQN Fraud Detection API!"}
