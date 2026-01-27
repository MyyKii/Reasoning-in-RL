import gymnasium as gym
import numpy as np
import json
import os
import logging

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------

DATA_PATH = "data/mlp_training_data.json"
ENV_NAME = "InvertedPendulum-v4"

NUM_EPISODES = 10000
MAX_STEPS = 500

LABEL_MODE = "binary"  # "binary" or "continuous"

THRESHOLD_THETA = 0.2
THRESHOLD_THETA_DOT = 1.2

logger = logging.getLogger(__name__)

def compute_label_from_state(state: np.ndarray) -> float:
    """
    Label = |theta_dot|, bei 1.0 abgeschnitten.
    theta_dot ist der 4. Eintrag im Zustandsvektor (Index 3).
    """
    theta_dot = float(state[3])
    return min(1.0, abs(theta_dot))


def compute_label(obs: np.ndarray, mode: str, terminated: bool | None = None) -> float:
    """
    InvertedPendulum Observation order (Gymnasium):
      [x, theta, x_dot, theta_dot]
    """
    x, theta, x_dot, theta_dot = [float(v) for v in obs[:4]]

    if mode == "binary":
        # Am saubersten: terminated als Ground-Truth nehmen (env definiert unhealthy über |theta|>0.2).
        if terminated is not None:
            return 1.0 if bool(terminated) else 0.0
        # Fallback:
        return 1.0 if abs(theta) > THRESHOLD_THETA else 0.0

    elif mode == "continuous":
        # Normiertes Risiko in [0,1] (hilft enorm, damit risk_plus/risk_minus NICHT flach werden)
        risk_theta = min(1.0, abs(theta) / THRESHOLD_THETA)           # THRESHOLD_THETA=0.2 passt zur Env-Definition
        risk_tdot  = min(1.0, abs(theta_dot) / THRESHOLD_THETA_DOT)   # Skala heuristisch
        return float(max(risk_theta, 0.25 * risk_tdot))

    else:
        raise ValueError(f"Unbekannter LABEL_MODE: {mode}")


def run_mlp_data_collection() -> None:
    """
    Einstiegspunkt für die Pipeline:
    - erstellt die Environment
    - startet W&B
    - ruft die eigentliche Sammel-Funktion auf
    - schließt die Environment wieder
    """
    logger.info(
        f"Starte MLP Data Collection: env={ENV_NAME}, "
        f"episodes={NUM_EPISODES}, max_steps={MAX_STEPS}, label_mode={LABEL_MODE}"
    )

    # W&B-Run initialization
    config = {
        "env_name": ENV_NAME,
        "num_episodes": NUM_EPISODES,
        "max_steps": MAX_STEPS,
        "label_mode": LABEL_MODE,
        "threshold_theta": THRESHOLD_THETA,
        "threshold_theta_dot": THRESHOLD_THETA_DOT,
    }

    init_wandb_run(
        project="counterfactual-agents",
        job_type="mlp_data_collection",
        config=config,
        run_name="mlp_data_collection",
    )

    env = gym.make(ENV_NAME, render_mode=None)
    try:
        collect_data(env)
    finally:
        env.close()
        finish_wandb_run()


def collect_data(env: gym.Env) -> None:
    """
    Führt die eigentliche Datensammlung durch:
    - Episoden durchlaufen
    - (state, action, label)-Tripel sammeln
    - JSON-Datei schreiben
    - Logging + W&B-Metriken
    """
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    data = []
    episode_rewards = []

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        done = False
        truncated = False

        episode_reward = 0.0
        episode_steps = 0

        while not (done or truncated) and episode_steps < MAX_STEPS:
            action_env = env.action_space.sample()


            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            if isinstance(action_env, np.ndarray):
                if action_env.size == 1:
                    action_value = float(action_env.item())   
                else:
                    action_value = action_env.tolist()      
            else:
                action_value = float(action_env)

            # 4) Label aus aktuellem Zustand berechnen (wie in deiner Referenz-JSON)
            #label = compute_label_from_state(obs)
            label = compute_label(next_obs, LABEL_MODE, terminated=terminated)


            # 5) Save Data
            data.append(
                {
                    "state": obs.tolist(),
                    "action": action_value,
                    "next_state": next_obs.tolist(),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "label": float(label),
                }
            )


            obs = next_obs
            episode_reward += reward
            episode_steps += 1

        episode_rewards.append(episode_reward)

        if (ep + 1) % 100 == 0 or ep == 0:
            logger.info(
                f"Episode {ep+1}/{NUM_EPISODES} abgeschlossen: "
                f"Reward={episode_reward:.2f}, Steps={episode_steps}"
            )


        log_metrics(
            {
                "episode_reward": float(episode_reward),
                "episode_length": episode_steps,
            },
            step=ep,
        )


    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)

    labels = [d["label"] for d in data]
    num_points = len(labels)

    logger.info(f"{num_points} Datenpunkte gespeichert in {DATA_PATH}")

    label_array = np.array(labels, dtype=float)
    label_min = float(label_array.min())
    label_max = float(label_array.max())
    label_mean = float(label_array.mean())

    logger.info(
        f"Kontinuierliche Labels: min={label_min:.3f}, "
        f"max={label_max:.3f}, mean={label_mean:.3f}"
    )

    log_metrics(
        {
            "num_samples": num_points,
            "label_min": label_min,
            "label_max": label_max,
            "label_mean": label_mean,
        }
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_mlp_data_collection()
