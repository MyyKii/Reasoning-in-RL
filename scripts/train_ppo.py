from stable_baselines3 import PPO
import gymnasium as gym

PPO_MODEL_PATH = "models/ppo_invertedpendulum.zip"

def train_ppo():
    env = gym.make("InvertedPendulum-v4")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=1_000_000)
    model.save(PPO_MODEL_PATH)

if __name__ == "__main__":
    train_ppo()