import gymnasium as gym
import numpy as np
from Anfis.TSFuzzyController import TSFuzzyController

def collect_data(episodes=10, steps=300, render=False):
    env = gym.make("InvertedPendulum-v4", render_mode="human" if render else None)
    controller = TSFuzzyController()
    
    X = []  # Eingaben: [theta, theta_dot]
    Y = []  # Zielwerte: Kraft (vom TS-Fuzzy-Controller)

    for ep in range(episodes):
        obs, _ = env.reset()
        for _ in range(steps):
            theta = obs[2]
            theta_dot = obs[3]

            u = controller.compute(theta, theta_dot)

            X.append([theta, theta_dot])
            Y.append(u)

            obs, _, terminated, truncated, _ = env.step([u])
            if terminated or truncated:
                break

    env.close()
    return np.array(X), np.array(Y)
