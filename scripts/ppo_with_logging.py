"""
ppo_with_logging.py

PPO Baseline (Train optional) + Evaluation mit W&B Logging
im gleichen Metric-Schema wie anfis_live_run.py.

Beispiele:

# Eval-only (wenn Modell existiert)
python scripts/ppo_with_logging.py \
  --env-id InvertedPendulum-v4 \
  --model-path models/ppo_invertedpendulum.zip \
  --eval-only \
  --episodes 20 \
  --max-steps 5000 \
  --seed 0 \
  --deterministic \
  --wandb-project counterfactual-agents \
  --run-name ppo_eval_seed0

# Train + Eval
python scripts/ppo_with_logging.py \
  --env-id InvertedPendulum-v4 \
  --model-path models/ppo_invertedpendulum.zip \
  --total-timesteps 100000 \
  --episodes 20 \
  --max-steps 5000 \
  --seed 0 \
  --deterministic \
  --wandb-project counterfactual-agents \
  --run-name ppo_train_eval_seed0
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

logger = logging.getLogger(__name__)


def evaluate_and_log(
    model: PPO,
    env_id: str,
    episodes: int,
    max_steps: int,
    seed: Optional[int],
    deterministic: bool,
    log_steps: bool,
    run_active: bool,
) -> Tuple[float, float, float]:
    env = gym.make(env_id)
    action_low = np.asarray(env.action_space.low, dtype=float).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=float).reshape(-1)

    global_step = 0
    returns: list[float] = []
    lengths: list[int] = []
    start_t = time.time()

    try:
        for ep in range(episodes):
            ep_seed = None if seed is None else int(seed) + int(ep)
            obs, _info = env.reset(seed=ep_seed)
            obs = np.asarray(obs, dtype=float)

            ep_ret = 0.0
            ep_len = 0

            for _ in range(max_steps):
                action, _ = model.predict(obs, deterministic=deterministic)
                action = np.asarray(action, dtype=float).reshape(-1)
                action = np.clip(action, action_low, action_high)

                next_obs, reward, terminated, truncated, _info = env.step(action)
                done = bool(terminated or truncated)

                ep_ret += float(reward)
                ep_len += 1
                global_step += 1

                if run_active and log_steps:
                    log_metrics(
                        {
                            "step/reward": float(reward),
                            "step/action": float(action.reshape(-1)[0]),
                            "step/episode": ep,
                        },
                        step=global_step,
                    )

                obs = np.asarray(next_obs, dtype=float)
                if done:
                    break

            returns.append(ep_ret)
            lengths.append(ep_len)
            logger.info("Episode %d/%d | return=%.3f | len=%d", ep + 1, episodes, ep_ret, ep_len)

            # WICHTIG: pro Episode loggen -> dann bekommst du eine Kurve statt nur einen Punkt
            if run_active:
                # Für saubere X-Achse: step=ep (Episode-Index)
                log_metrics(
                    {"episode/return": ep_ret, "episode/length": ep_len, "episode/index": ep},
                    step=ep,
                )

        dur = time.time() - start_t
        mean_ret = float(np.mean(returns)) if returns else float("nan")
        std_ret = float(np.std(returns)) if returns else float("nan")
        mean_len = float(np.mean(lengths)) if lengths else float("nan")

        logger.info(
            "Done. mean_return=%.3f ± %.3f | mean_len=%.1f | runtime=%.1fs",
            mean_ret, std_ret, mean_len, dur
        )

        if run_active:
            # Summary kann am Ende einmalig geloggt werden
            log_metrics(
                {
                    "summary/mean_return": mean_ret,
                    "summary/std_return": std_ret,
                    "summary/mean_length": mean_len,
                    "summary/runtime_sec": float(dur),
                },
                step=episodes,  # optional, nur damit es nicht bei step=0 hängt
            )

        return mean_ret, std_ret, mean_len
    finally:
        env.close()


def train_and_or_eval(
    env_id: str,
    model_path: str,
    total_timesteps: int,
    eval_only: bool,
    episodes: int,
    max_steps: int,
    seed: Optional[int],
    deterministic: bool,
    log_steps: bool,
    wandb_project: Optional[str],
    run_name: Optional[str],
) -> None:
    if seed is not None:
        set_random_seed(int(seed))

    run_active = False
    if wandb_project:
        cfg = {
            "algo": "PPO",
            "env_id": env_id,
            "model_path": model_path,
            "total_timesteps": int(total_timesteps),
            "eval_only": bool(eval_only),
            "episodes": int(episodes),
            "max_steps": int(max_steps),
            "seed": seed,
            "deterministic": bool(deterministic),
            "log_steps": bool(log_steps),
        }
        init_wandb_run(
            project=wandb_project,
            job_type="ppo_eval" if eval_only else "ppo_train_eval",
            config=cfg,
            run_name=run_name,
        )
        run_active = True

    try:
        model_file = Path(model_path)

        if eval_only:
            if not model_file.exists():
                raise FileNotFoundError(f"PPO model not found: {model_path}")
            logger.info("Loading PPO model: %s", model_path)
            model = PPO.load(model_path)
        else:
            logger.info("Training PPO: env=%s timesteps=%d seed=%s", env_id, total_timesteps, str(seed))
            env = gym.make(env_id)
            try:
                env.reset(seed=seed)
                model = PPO("MlpPolicy", env, verbose=1, seed=seed)
                model.learn(total_timesteps=int(total_timesteps))
            finally:
                env.close()

            model_file.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Saving PPO model to: %s", model_path)
            model.save(model_path)

        evaluate_and_log(
            model=model,
            env_id=env_id,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            deterministic=deterministic,
            log_steps=log_steps,
            run_active=run_active,
        )
    finally:
        if run_active:
            finish_wandb_run()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="PPO Baseline mit W&B Logging (Episode/Summary kompatibel zu ANFIS Live-Run)")
    ap.add_argument("--env-id", default="InvertedPendulum-v4")
    ap.add_argument("--model-path", default="models/ppo_invertedpendulum.zip")
    ap.add_argument("--total-timesteps", type=int, default=100_000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-steps", action="store_true", help="Optional: step/* wie bei ANFIS loggen (mehr Daten)")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    train_and_or_eval(
        env_id=args.env_id,
        model_path=args.model_path,
        total_timesteps=args.total_timesteps,
        eval_only=args.eval_only,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        deterministic=args.deterministic,
        log_steps=args.log_steps,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
