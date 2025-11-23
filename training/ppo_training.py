import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from environment.custom_env import FraudDetectionEnv

env = FraudDetectionEnv()
model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, batch_size=32, gamma=0.99)

model.learn(total_timesteps=10000)
os.makedirs("models", exist_ok=True)
model.save("models/ppo_fraud_model")
print("PPO training complete and model saved.")
