import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from stable_baselines3 import DQN, PPO, A2C
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import FraudDetectionEnv

# -------------------------------
# Evaluation function for PyTorch REINFORCE model
# -------------------------------
def evaluate_reinforce(model_path, env, episodes=1000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    class PolicyNet(nn.Module):
        def __init__(self, state_dim, action_dim):
            super(PolicyNet, self).__init__()
            self.fc = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
                nn.Softmax(dim=-1)
            )
        def forward(self, x):
            return self.fc(x)

    device = torch.device("cpu")
    policy = PolicyNet(state_dim, action_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    y_true, y_pred, rewards = [], [], []

    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = policy(state_tensor).squeeze(0)
            m = torch.distributions.Categorical(probs)
            action = m.sample().item()

            true_label = int(env.current_transaction[-1])
            y_true.append(true_label)
            y_pred.append(action)

            state, reward, terminated, _, _ = env.step(action)
            episode_reward += reward
            done = terminated
        rewards.append(episode_reward)

    return y_true, y_pred, rewards

# -------------------------------
# Evaluation function for Stable-Baselines models
# -------------------------------
def evaluate_sb3(model, env, episodes=1000):
    y_true, y_pred, rewards = [], [], []
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action, _ = model.predict(state)
            true_label = int(env.current_transaction[-1])
            y_true.append(true_label)
            y_pred.append(action)
            state, reward, terminated, _, _ = env.step(action)
            episode_reward += reward
            done = terminated
        rewards.append(episode_reward)
    return y_true, y_pred, rewards

# -------------------------------
# Initialize environment
# -------------------------------
env = FraudDetectionEnv()
episodes = 1000

# -------------------------------
# Load all models
# -------------------------------
models = {
    "DQN": DQN.load("models/dqn_fraud_model"),
    "REINFORCE": "models/reinforce_fraud_model.pt",
    "PPO": PPO.load("models/ppo_fraud_model"),
    "A2C": A2C.load("models/a2c_fraud_model")
}

results = {}

# -------------------------------
# Evaluate each model
# -------------------------------
for name, model in models.items():
    print(f"\nEvaluating {name}...")
    if name == "REINFORCE":
        y_true, y_pred, rewards = evaluate_reinforce(model, env, episodes)
    else:
        y_true, y_pred, rewards = evaluate_sb3(model, env, episodes)

    # Metrics
    accuracy = np.mean([y_true[i] == y_pred[i] for i in range(len(y_true))])
    total_frauds = sum(y_true)
    detected_frauds = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
    fraud_acc = detected_frauds / max(1, total_frauds)
    avg_reward = np.mean(rewards)

    results[name] = {
        "y_true": y_true,
        "y_pred": y_pred,
        "rewards": rewards,
        "accuracy": accuracy,
        "fraud_acc": fraud_acc,
        "avg_reward": avg_reward,
        "detected_frauds": detected_frauds,
        "total_frauds": total_frauds
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title(f"{name} Confusion Matrix")
    plt.show()

    # Reward plot
    fig2, ax2 = plt.subplots()
    ax2.plot(rewards)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title(f"{name} Reward per Episode")
    ax2.grid(True)
    plt.show()

# -------------------------------
# Comparison table
# -------------------------------
print("\n===== MODEL COMPARISON =====")
print(f"{'Algorithm':<12} {'Accuracy':<10} {'Fraud Acc':<12} {'Avg Reward':<10} {'Detected/Total Fraud':<20}")
for name, metrics in results.items():
    print(f"{name:<12} {metrics['accuracy']*100:>7.2f}% {metrics['fraud_acc']*100:>10.2f}% "
          f"{metrics['avg_reward']:>10.2f} {metrics['detected_frauds']}/{metrics['total_frauds']}")
