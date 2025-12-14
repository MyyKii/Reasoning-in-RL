"""anfis_live_run.py

Lädt ein gespeichertes lazuardy-ANFIS-Bundle und nutzt es als Controller
für einen "Live-Run" im MuJoCo InvertedPendulum.

Erwartet ein Bundle, das mit anfis_model.py via --bundle-out geschrieben wurde:
  <bundle>.model.pkl  (ANFIS Modell)
  <bundle>.bundle.pkl (Preprocess + y_stats + meta)

Beispiel:
  python scripts/anfis_live_run.py \
    --env-id InvertedPendulum-v4 \
    --bundle models/anfis_controller \
    --episodes 10 \
    --max-steps 1000 \
    --render \
    --wandb-project counterfactual-agents \
    --run-name anfis_live
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

try:
    import gymnasium as gym
    _GYMNASIUM = True
except Exception:
    import gym  # type: ignore
    _GYMNASIUM = False


# Repo-Struktur tolerant halten
try:
    from utils.anfis_io import load_anfis_bundle, transform_X_with
except ModuleNotFoundError:
    from anfis_io import load_anfis_bundle, transform_X_with

try:
    from logging_config import setup_logging
except ModuleNotFoundError:
    setup_logging = None  # optional

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run


def _predict_action(
    model,
    preprocess: dict,
    y_stats: dict,
    obs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """Berechnet a_t = denorm(ANFIS(norm(obs))) und clippt auf action space."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    if obs.shape[0] < 4:
        raise ValueError(f"Observation hat zu wenige Dimensionen: {obs.shape}")

    # erwartetes Feature-Layout: [x, theta, x_dot, theta_dot]
    x = obs[:4].reshape(1, 4)
    x_n = transform_X_with(preprocess, x)

    # lazuardy_anfis liefert typischerweise shape (N,1) oder (N,)
    y_n = np.asarray(model.predict(x_n)).reshape(-1)[0]
    a = y_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])

    a_arr = np.array([a], dtype=float)
    a_arr = np.clip(a_arr, action_low, action_high)
    return a_arr


def run_live(
    env_id: str,
    bundle_base: Path,
    episodes: int,
    max_steps: int,
    seed: int | None,
    render: bool,
    wandb_project: str | None,
    run_name: str | None,
) -> None:
    # --- Logging ---
    if setup_logging is not None:
        setup_logging(log_file="anfis_live_run.log")
    else:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Load ANFIS bundle ---
    model, preprocess, y_stats, meta = load_anfis_bundle(bundle_base)
    logging.info("Loaded ANFIS bundle from %s(.model.pkl/.bundle.pkl)", bundle_base)
    logging.info("Bundle meta: %s", meta)

    # --- Env ---
    make_kwargs = {}
    if render:
        make_kwargs["render_mode"] = "human"  # gymnasium

    env = gym.make(env_id, **make_kwargs)
    action_low = np.asarray(env.action_space.low, dtype=float).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=float).reshape(-1)

    # --- W&B (optional) ---
    run = None
    if wandb_project:
        cfg = {
            "env_id": env_id,
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
            "bundle_base": str(bundle_base),
            "bundle_meta": meta,
        }
        run = init_wandb_run(project=wandb_project, job_type="anfis_live_run", config=cfg, run_name=run_name)

    try:
        global_step = 0
        returns = []
        lengths = []
        start_t = time.time()

        for ep in range(episodes):
            ep_seed = None if seed is None else int(seed) + int(ep)
            if _GYMNASIUM:
                obs, _info = env.reset(seed=ep_seed)
            else:
                obs = env.reset()
            obs = np.asarray(obs, dtype=float)

            ep_ret = 0.0
            ep_len = 0

            for _ in range(max_steps):
                action = _predict_action(model, preprocess, y_stats, obs, action_low, action_high)

                step_out = env.step(action)
                if _GYMNASIUM and len(step_out) == 5:
                    next_obs, reward, terminated, truncated, _info = step_out
                    done = bool(terminated or truncated)
                else:
                    next_obs, reward, done, _info = step_out

                ep_ret += float(reward)
                ep_len += 1
                global_step += 1

                if run is not None:
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

            logging.info("Episode %d/%d | return=%.3f | len=%d", ep + 1, episodes, ep_ret, ep_len)
            if run is not None:
                log_metrics(
                    {"episode/return": ep_ret, "episode/length": ep_len, "episode/index": ep},
                    step=global_step,
                )

        dur = time.time() - start_t
        mean_ret = float(np.mean(returns)) if returns else float("nan")
        std_ret = float(np.std(returns)) if returns else float("nan")
        mean_len = float(np.mean(lengths)) if lengths else float("nan")
        logging.info("Done. mean_return=%.3f ± %.3f | mean_len=%.1f | runtime=%.1fs", mean_ret, std_ret, mean_len, dur)

        if run is not None:
            log_metrics(
                {
                    "summary/mean_return": mean_ret,
                    "summary/std_return": std_ret,
                    "summary/mean_length": mean_len,
                    "summary/runtime_sec": float(dur),
                },
                step=global_step,
            )
    finally:
        env.close()
        if run is not None:
            finish_wandb_run()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ANFIS Controller Live-Run im MuJoCo InvertedPendulum")
    ap.add_argument("--env-id", default="InvertedPendulum-v4")
    ap.add_argument("--bundle", type=Path, required=True, help="Basispfad ohne Suffix, z.B. models/anfis_controller")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    run_live(
        env_id=args.env_id,
        bundle_base=args.bundle,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
