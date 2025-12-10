from stable_baselines3 import PPO
import gymnasium as gym

Model_PATH = "models/ppo_invertedpendulum.zip"

env = gym.make("InvertedPendulum-v4")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
model.save(Model_PATH)
