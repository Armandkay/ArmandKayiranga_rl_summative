import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from stable_baselines3 import DQN
from environment.custom_env import FraudDetectionEnv
from environment.rendering import render_transaction

# -------------------------------
# 1. Load trained DQN model
# -------------------------------
model_path = "models/dqn_fraud_model"
model = DQN.load(model_path)
env = FraudDetectionEnv()

# -------------------------------
# 2. Evaluation parameters
# -------------------------------
episodes = 1000
correct = 0
fraud_count = 0
fraud_detected = 0
all_rewards = []

y_true = []
y_pred = []

# -------------------------------
# 3. Run evaluation loop
# -------------------------------
for ep in range(episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(state)
        true_label = int(env.current_transaction[-1])

        # Track metrics
        if action == true_label:
            correct += 1

        if true_label == 1:
            fraud_count += 1
            if action == 1:
                fraud_detected += 1

        y_true.append(true_label)
        y_pred.append(action)

        state, reward, terminated, _, info = env.step(action)
        episode_reward += reward
        done = terminated

    all_rewards.append(episode_reward)

# -------------------------------
# 4. Print results
# -------------------------------
accuracy = correct / episodes
fraud_accuracy = fraud_detected / max(fraud_count, 1)

print(f"--- DQN Evaluation ---")
print(f"Total Episodes: {episodes}")
print(f"Overall Accuracy: {accuracy*100:.2f}%")
print(f"Fraud Detection Accuracy: {fraud_accuracy*100:.2f}%")
print(f"Average Reward per Episode: {np.mean(all_rewards):.2f}")

# -------------------------------
# 5. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
disp.plot(cmap=plt.cm.Blues)
plt.title("DQN Confusion Matrix")
plt.show()

# -------------------------------
# 6. Reward per episode plot
# -------------------------------
plt.figure(figsize=(10,5))
plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("DQN Reward per Episode")
plt.grid(True)
plt.show()

# -------------------------------
# 7. Optional: visualize a few transactions
# -------------------------------
print("\nSample transaction predictions:")
for _ in range(5):
    state, _ = env.reset()
    action, _ = model.predict(state)
    render_transaction(env.current_transaction, action)
