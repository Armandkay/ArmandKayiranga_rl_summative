import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import sys
import os

# -------------------------------
# Fix for imports
# -------------------------------
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment.custom_env import FraudDetectionEnv

# -------------------------------
# Load DQN model and environment
# -------------------------------
model = DQN.load("models/dqn_fraud_model")
env = FraudDetectionEnv()

# Initialize lists to avoid NameError
y_true = []
y_pred = []
rewards = []

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("DQN Fraud Detection Simulator")
st.write("Enter transaction details below to see the agent's prediction:")

# Input form
with st.form(key="transaction_form"):
    amount = st.number_input("Transaction Amount ($)", min_value=1.0, max_value=10000.0, value=500.0)
    hour = st.slider("Hour of Transaction", min_value=0, max_value=23, value=12)
    location = st.number_input("Location Code", min_value=0, max_value=100, value=0)
    risk_score = st.slider("Risk Score", min_value=0.0, max_value=1.0, value=0.5)
    submit_button = st.form_submit_button(label="Predict")

# Predict transaction
if submit_button:
    obs = np.array([amount, hour, location, risk_score], dtype=np.float32)
    env.current_transaction = np.append(obs, 0)  # dummy label
    action, _ = model.predict(obs)
    label = "Fraud" if action == 1 else "Legit"
    st.success(f"Prediction: **{label}**")

# -------------------------------
# Batch simulation and visualization
# -------------------------------
st.header("Batch Simulation of 100 Transactions")

if st.button("Run Simulation"):
    episodes = 100
    y_true = []
    y_pred = []
    rewards = []

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

    # Accuracy
    accuracy = np.mean([y_true[i] == y_pred[i] for i in range(len(y_true))])
    st.write(f"Overall Accuracy: {accuracy*100:.2f}%")
    fraud_acc = sum(1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1) / max(1, sum(y_true))
    st.write(f"Fraud Detection Accuracy: {fraud_acc*100:.2f}%")
    st.write(f"Average Reward per Episode: {np.mean(rewards):.2f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    st.pyplot(fig)

    # Reward plot
    fig2, ax2 = plt.subplots()
    ax2.plot(rewards)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("Reward per Episode")
    ax2.grid(True)
    st.pyplot(fig2)

# Optional: show raw data
if st.checkbox("Show raw predictions table"):
    if y_true and y_pred:
        df = pd.DataFrame({"True Label": y_true, "Predicted Label": y_pred})
        st.dataframe(df)
    else:
        st.write("Run batch simulation first.")
