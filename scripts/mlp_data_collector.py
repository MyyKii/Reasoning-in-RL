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

# "binary" or "continuous"
LABEL_MODE = "continuous"

# Thresholds für Labels
THRESHOLD_THETA = 0.15
THRESHOLD_THETA_DOT = 1.0

logger = logging.getLogger(__name__)

def compute_label_from_state(state: np.ndarray) -> float:
    """
    Label = |theta_dot|, bei 1.0 abgeschnitten.
    theta_dot ist der 4. Eintrag im Zustandsvektor (Index 3).
    """
    theta_dot = float(state[3])
    return min(1.0, abs(theta_dot))


def compute_label(obs: np.ndarray, mode: str) -> float:
    """
    Berechnet ein Label aus der Beobachtung.
    Hier ein Beispiel:
      - binary: 1 wenn "riskant", sonst 0
      - continuous: kontinuierliches Maß für "Gefährlichkeit"

    Hinweis: Für InvertedPendulum ist die Beobachtung typischerweise:
      [x, x_dot, theta, theta_dot]
    Das musst du ggf. mit deiner Umgebung abgleichen.
    """
    # Beispiel: wir nehmen an, Index 2 = theta, Index 3 = theta_dot
    theta = obs[2]
    theta_dot = obs[3]

    if mode == "binary":
        risky = (abs(theta) > THRESHOLD_THETA) or (abs(theta_dot) > THRESHOLD_THETA_DOT)
        return 1.0 if risky else 0.0

    elif mode == "continuous":
        # Beispiel: je weiter über dem Threshold, desto höher das Label
        theta_score = max(0.0, abs(theta) - THRESHOLD_THETA)
        theta_dot_score = max(0.0, abs(theta_dot) - THRESHOLD_THETA_DOT)
        # einfache Kombination
        return float(theta_score + 0.1 * theta_dot_score)

    else:
        raise ValueError(f"Unbekannter LABEL_MODE: {mode}")


# ---------------------------------------------------------
# main Pipeline
# ---------------------------------------------------------

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
            # 1) zufällige Action für die Umgebung
            action_env = env.action_space.sample()

            # 2) Schritt ausführen
            next_obs, reward, terminated, truncated, info = env.step(action_env)
            done = terminated or truncated

            # 3) Action als Skalar für die JSON extrahieren
            if isinstance(action_env, np.ndarray):
                if action_env.size == 1:
                    action_value = float(action_env.item())   # Skalar z.B. 0.27
                else:
                    action_value = action_env.tolist()        # (für spätere multidim Actions)
            else:
                action_value = float(action_env)

            # 4) Label aus aktuellem Zustand berechnen (wie in deiner Referenz-JSON)
            label = compute_label_from_state(obs)

            # 5) Datensatz-Eintrag speichern
            data.append(
                {
                    "state": obs.tolist(),       # [x, x_dot, theta, theta_dot]
                    "action": action_value,      # Skalar
                    "label": float(label),       # Skalar
                }
            )

            # 6) Zustand & Zähler updaten
            obs = next_obs
            episode_reward += reward
            episode_steps += 1

        episode_rewards.append(episode_reward)

        # Logging
        if (ep + 1) % 100 == 0 or ep == 0:
            logger.info(
                f"Episode {ep+1}/{NUM_EPISODES} abgeschlossen: "
                f"Reward={episode_reward:.2f}, Steps={episode_steps}"
            )

        # Logging für W&B
        log_metrics(
            {
                "episode_reward": float(episode_reward),
                "episode_length": episode_steps,
            },
            step=ep,
        )

    # JSON-Datei speichern
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f)

    labels = [d["label"] for d in data]
    num_points = len(labels)

    logger.info(f"{num_points} Datenpunkte gespeichert in {DATA_PATH}")

    # Zusammenfassung für Labels (hier nur kontinuierlich)
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
