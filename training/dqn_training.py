from stable_baselines3 import DQN
from environment.custom_env import FraudDetectionEnv

def train_dqn(total_timesteps=20000):
    env = FraudDetectionEnv()
    model = DQN('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=total_timesteps)
    model.save("models/dqn_fraud_model")
    print("DQN Fraud Detection training complete!")

if __name__ == "__main__":
    train_dqn()
