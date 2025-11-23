import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from environment.custom_env import FraudDetectionEnv

# -------------------------------
# Policy network
# -------------------------------
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

# -------------------------------
# Hyperparameters
# -------------------------------
lr = 0.001
gamma = 0.99
episodes = 1000

env = FraudDetectionEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

policy = PolicyNet(state_dim, action_dim)
optimizer = optim.Adam(policy.parameters(), lr=lr)

# -------------------------------
# Training loop
# -------------------------------
for ep in range(episodes):
    state, _ = env.reset()
    rewards = []
    log_probs = []

    done = False
    while not done:
        # Add batch dimension for policy
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # shape [1, state_dim]
        probs = policy(state_tensor).squeeze(0)  # shape [action_dim]
        
        m = torch.distributions.Categorical(probs)
        action = m.sample()
        log_prob = m.log_prob(action)
        
        # Take step in environment
        state, reward, terminated, _, _ = env.step(action.item())
        rewards.append(reward)
        log_probs.append(log_prob)
        done = terminated

    # Compute returns
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns = torch.tensor(returns, dtype=torch.float32)

    # Normalize returns if more than 1 step
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

    # Policy gradient update
    loss = 0
    for log_prob, Gt in zip(log_probs, returns):
        loss += -log_prob * Gt
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (ep+1) % 100 == 0:
        print(f"Episode {ep+1}/{episodes}, total reward: {sum(rewards)}")

# Save model
os.makedirs("models", exist_ok=True)
torch.save(policy.state_dict(), "models/reinforce_fraud_model.pt")
print("REINFORCE training complete and model saved.")
