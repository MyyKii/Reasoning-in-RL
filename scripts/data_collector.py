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

def compute_label(state, threshold_theta=0.15, threshold_theta_dot=1.0):   #threshold_theta=0.2
    # Heuristik: risky, wenn zu viel Winkel oder Winkelgeschwindigkeit
    theta = state[1]   # in InvertedPendulum-v4: [x, theta, x_dot, theta_dot]
    theta_dot = state[3]
    return int(abs(theta) > threshold_theta or abs(theta_dot) > threshold_theta_dot) #label 0: safe, 1: risky

def collect_data(env, num_episodes=50, max_steps=200):
    data = load_data()
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        for step in range(max_steps):
            # Aktion: Zufällig oder heuristisch
            action = env.action_space.sample() 
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Format: [x, theta, x_dot, theta_dot]
            state = next_obs.tolist()
            act = float(action[0])  

            # Automatisches Label (kannst du auch durch input() ersetzen)
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
    print(f"Geladene Datenpunkte: {len(data)}")


    thetas = [d["state"][1] for d in data]
    count_over = sum(abs(t) > 0.2 for t in thetas)
    print(f"|theta| > 0.2 in {count_over} von {len(thetas)} Samples "
        f"({count_over/len(thetas):.2%})")


if __name__ == "__main__":
    env = gym.make("InvertedPendulum-v4", render_mode=None)  # optional: render_mode="human"
    collect_data(env)
    env.close()
