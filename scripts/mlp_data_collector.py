import gymnasium as gym
import numpy as np
import json
import os

# config
DATA_PATH = "collected_data.json"
NUM_EPISODES = 10000
MAX_STEPS = 500
LABEL_MODE = "continuos"  # "binary" or "continuous"

# Thresholds for labels
THRESHOLD_THETA = 0.15
THRESHOLD_THETA_DOT = 1.0


def load_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

def compute_binary_label(state):
    """0 = safe, 1 = risky"""
    theta = state[1]
    theta_dot = state[3]
    return int(abs(theta) > THRESHOLD_THETA or abs(theta_dot) > THRESHOLD_THETA_DOT)

def compute_continuous_label(state):
    """Risikowert zwischen 0 und 1"""
    theta = state[1]
    theta_dot = state[3]
    risk_theta = min(1.0, abs(theta)/0.2)
    risk_theta_dot = min(1.0, abs(theta_dot)/1.0)
    return max(risk_theta, risk_theta_dot)

def compute_label(state):
    if LABEL_MODE == "binary":
        return compute_binary_label(state)
    else:
        return compute_continuous_label(state)

def collect_data(env, num_episodes=NUM_EPISODES, max_steps=MAX_STEPS):
    data = load_data()

    for episode in range(num_episodes):
        obs, _ = env.reset()
        for step in range(max_steps):
            action = env.action_space.sample()

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            state = next_obs.tolist()
            act = float(action[0])
            label = compute_label(state)

            data.append({
                "state": state,
                "action": act,
                "label": label
            })

            obs = next_obs
            if done:
                break

    save_data(data)

    print(f"{len(data)} Datenpunkte gespeichert in {DATA_PATH}")
    labels = [d["label"] for d in data]
    if LABEL_MODE == "binary":
        num_risky = sum(labels)
        print(f"Risky Samples: {num_risky} ({num_risky/len(labels):.2%})")
    else:
        print(f"Kontinuäre Labels: min={min(labels):.3f}, max={max(labels):.3f}, mean={np.mean(labels):.3f}")


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4", render_mode=None)  # render_mode="human" optional
    collect_data(env)
    env.close()
