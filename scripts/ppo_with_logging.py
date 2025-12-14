import logging
from typing import Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

MODEL_PATH = "models/ppo_invertedpendulum.zip"
TIMESTEPS = 100_000
EPISODES_EVAL = 10

logger = logging.getLogger(__name__)

def train_ppo_inverted_pendulum(
    total_timesteps: int = 100_000,
    n_eval_episodes: int = 10,
) -> Tuple[PPO, float, float]:
    """
    Trainiert ein PPO-Modell auf InvertedPendulum-v4,
    speichert es und loggt eine kurze Evaluation zu WandB.
    """
    # === Weights & Biases Run initialisieren ===
    run = init_wandb_run(
        project="counterfactual-agents",   # genau wie im MLP-Skript
        job_type="ppo_training",           # oder wie du es nennen willst
        config={
            "algo": "PPO",
            "env_id": "InvertedPendulum-v4",
            "total_timesteps": total_timesteps,
            "policy": "MlpPolicy",
        },
        run_name="ppo_baseline",
    )

    logger.info("WandB-Run gestartet: %s", getattr(run, "name", "unbekannt"))

    env = gym.make("InvertedPendulum-v4")
    model = PPO("MlpPolicy", env, verbose=1)

    logger.info("Starte PPO-Training (timesteps=%d)...", total_timesteps)
    model.learn(total_timesteps=total_timesteps)
    logger.info("PPO-Training abgeschlossen.")

    logger.info("Speichere PPO-Modell nach '%s'...", MODEL_PATH)
    model.save(MODEL_PATH)
    logger.info("Modell gespeichert.")

    # === einfache Evaluation zum Vergleich / Logging ===
    eval_env = gym.make("InvertedPendulum-v4")
    returns = []

    for ep in range(n_eval_episodes):
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

    logger.info(
        "PPO Evaluation über %d Episoden: Return = %.3f ± %.3f",
        n_eval_episodes,
        mean_return,
        std_return,
    )

    logger.info("Logge PPO-Metriken zu WandB...")
    log_metrics(
        {
            "ppo_mean_return": mean_return,
            "ppo_std_return": std_return,
        }
    )

    finish_wandb_run()

    env.close()
    eval_env.close()

    return model, mean_return, std_return



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_ppo_inverted_pendulum(TIMESTEPS, EPISODES_EVAL)