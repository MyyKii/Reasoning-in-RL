import gymnasium as gym
import numpy as np
import json
import os

DATA_PATH = "collected_data.json"

def load_data():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, "r") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_PATH, "w") as f:
        json.dump(data, f, indent=2)

def compute_label(state, threshold_theta=0.2, threshold_theta_dot=1.0):
    theta = state[1]
    theta_dot = state[3]
    return int(abs(theta) > threshold_theta or abs(theta_dot) > threshold_theta_dot)

def collect_data(env, num_episodes=5, max_steps=200):
    data = load_data()

    for episode in range(num_episodes):
        obs, _ = env.reset()

        # ---- Anpassung: Zufällige Startwerte setzen ----
        d = env.unwrapped.data

        d.qpos[1] = np.random.uniform(-0.4, 0.4)   # theta
        d.qvel[1] = np.random.uniform(-0.5, 0.5)   # theta_dot
        d.qpos[0] = np.random.uniform(-0.05, 0.05) # x
        d.qvel[0] = np.random.uniform(-0.05, 0.05) # x_dot

        d.forward()  # bei gymnasium+mujoco

        obs = np.concatenate([d.qpos, d.qvel])

        for step in range(max_steps):
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, _ = env.step(action)
            state = obs.tolist()
            act = float(action[0])
            label = compute_label(state)

            data.append({
                "state": state,
                "action": act,
                "label": label
            })

            obs = next_obs
            if terminated or truncated:
                break

    save_data(data)
    print(f"{len(data)} Datenpunkte gespeichert in {DATA_PATH}")

if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4")
    collect_data(env, num_episodes=20)
    env.close()
