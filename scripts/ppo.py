import logging
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

Model_PATH = "models/ppo_invertedpendulum.zip"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("ppo_training")

# === Weights & Biases Run initialisieren ===
run = init_wandb_run(
    project_name="inverted_pendulum",
    config={
        "algo": "PPO",
        "env_id": "InvertedPendulum-v4",
        "total_timesteps": 100_000,
        "policy": "MlpPolicy",
    },
    run_name="ppo_baseline"
)

env = gym.make("InvertedPendulum-v4")
model = PPO("MlpPolicy", env, verbose=1)

logger.info("Starte PPO-Training...")
model.learn(total_timesteps=100_000)
logger.info("PPO-Training abgeschlossen, speichere Modell...")
model.save(Model_PATH)
logger.info("Modell gespeichert unter %s", Model_PATH)

# === einfache Evaluation zum Loggen ===
eval_env = gym.make("InvertedPendulum-v4")
n_episodes = 10
returns = []

for ep in range(n_episodes):
    obs, _ = eval_env.reset()
    done = False
    ep_return = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated
        ep_return += reward
    returns.append(ep_return)

mean_return = float(np.mean(returns))
std_return = float(np.std(returns))

logger.info("Durchschnittlicher Return über %d Episoden: %.2f ± %.2f",
            n_episodes, mean_return, std_return)

log_metrics({
    "ppo_mean_return": mean_return,
    "ppo_std_return": std_return,
})

finish_wandb_run()
