````markdown
# AI-Powered Fraud Detection with Reinforcement Learning

## Project Overview
This project implements a **Reinforcement Learning system for real-time financial transaction fraud detection**. The goal is to classify transactions as **legitimate or fraudulent** using a custom environment and multiple RL algorithms (DQN, REINFORCE, PPO, A2C).  

## Environment
- **Observation Space:** Transaction features `[Amount, Hour, Location, Risk Score]`
- **Action Space:** `0 = Legitimate`, `1 = Fraud`
- **Reward:** +1 for correct, -1 for incorrect classification
- **Visualization:** Console print for transactions; Matplotlib/Streamlit for rewards and confusion matrices

## Algorithms Implemented
1. **DQN (Value-Based)**
2. **REINFORCE (Policy Gradient)**
3. **PPO (Proximal Policy Optimization)**
4. **A2C (Actor-Critic)**

## Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/Armandkay/ArmandKayiranga_rl_summative.git
cd <ArmandKayiranga_rl_summative>
````

### 2. Create & Activate Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train a Model

**Example: Train DQN**

```bash
python training/dqn_training.py
```

**Other Models**

* REINFORCE: `python training/reinforce_training.py`
* PPO: `python training/ppo_training.py`
* A2C: `python training/a2c_training.py`

### 5. Evaluate All Models

```bash
python evaluate_all_models.py
```

* Generates **comparison table**, **confusion matrices**, and **reward per episode plots**.

### 6. Run Visualization App (Optional)

```bash
streamlit run streamlit_app.py
```

* Allows interactive exploration of the environment and model predictions.

## Results

| Algorithm | Accuracy | Fraud Acc | Avg Reward | Detected/Total Fraud |
| --------- | -------- | --------- | ---------- | -------------------- |
| DQN       | 63.7%    | 1.68%     | 0.27       | 6/357                |
| REINFORCE | 63.7%    | 0%        | 0.27       | 0/363                |
| PPO       | 64.9%    | 0%        | 0.30       | 0/351                |
| A2C       | 63.8%    | 0%        | 0.28       | 0/362                |

## Contributing

Feel free to submit issues or pull requests to improve environment realism, model performance, or visualization.
