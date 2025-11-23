from environment.custom_env import FraudDetectionEnv
from environment.rendering import render_transaction

env = FraudDetectionEnv()
state, _ = env.reset()
done = False

while not done:
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    render_transaction(info["transaction"], action)
    done = terminated
