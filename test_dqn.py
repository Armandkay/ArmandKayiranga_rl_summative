from stable_baselines3 import DQN
from environment.custom_env import FraudDetectionEnv
from environment.rendering import render_transaction
import numpy as np

# Load the trained DQN model
model = DQN.load("models/dqn_fraud_model")
env = FraudDetectionEnv()

# Evaluation parameters
episodes = 1000
correct = 0
fraud_count = 0
fraud_detected = 0

for _ in range(episodes):
    state, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        true_label = int(env.current_transaction[-1])

        # Accuracy count
        if action == true_label:
            correct += 1

        # Fraud detection stats
        if true_label == 1:
            fraud_count += 1
            if action == 1:
                fraud_detected += 1

        # Optional: render some transactions
        # render_transaction(env.current_transaction, action)

        state, _, terminated, _, _ = env.step(action)
        done = terminated

accuracy = correct / episodes
fraud_accuracy = fraud_detected / max(fraud_count, 1)

print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Fraud Detection Accuracy: {fraud_accuracy*100:.2f}%")
