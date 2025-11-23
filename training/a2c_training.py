import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import A2C
from environment.custom_env import FraudDetectionEnv

env = FraudDetectionEnv()
model = A2C("MlpPolicy", env, verbose=1, learning_rate=0.001, gamma=0.99)

model.learn(total_timesteps=10000)
os.makedirs("models", exist_ok=True)
model.save("models/a2c_fraud_model")
print("A2C training complete and model saved.")
