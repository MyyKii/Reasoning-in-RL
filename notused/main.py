import gymnasium as gym
import numpy as np

# Environment erstellen
env = gym.make("InvertedPendulum-v4", render_mode="human")  

# Reset: Startzustand holen
observation, info = env.reset()

# 500 Schritte testen
for _ in range(500):
    action = np.array([0.0])  # keine Kraft – Pendel fällt um
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        print("Episode beendet. Starte neu.")
        observation, info = env.reset()

env.close() 
