"""counterfactual_live_run.py

Live-Run: ANFIS liefert Basisaktion a.
MLP bewertet Risk für (a-delta, a, a+delta).
Wir führen die Aktion mit minimalem Risk aus.

Beispiel:
python scripts/counterfactual_live_run.py \
  --env-id InvertedPendulum-v4 \
  --bundle models/anfis_controller_dagger1 \
  --mlp-model models/mlp_model.pth \
  --delta 0.15 \
  --episodes 20 \
  --max-steps 5000 \
  --seed 0 \
  --wandb-project counterfactual-agents \
  --run-name anfis_counterfactual_live
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

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

import mlp_model  # nutzt load_risk_mlp() und predict_risk()


def _predict_anfis_action(model, preprocess: dict, y_stats: dict, obs: np.ndarray,
                         action_low: np.ndarray, action_high: np.ndarray) -> np.ndarray:
    obs = np.asarray(obs, dtype=float).reshape(-1)
    x = obs[:4].reshape(1, 4)
    x_n = transform_X_with(preprocess, x)

    y_n = np.asarray(model.predict(x_n)).reshape(-1)[0]
    a = y_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])
    a_arr = np.array([a], dtype=float)
    return np.clip(a_arr, action_low, action_high)


def _choose_action_with_mlp_risk(
    risk_model,
    obs: np.ndarray,
    a0: float,
    delta: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> tuple[float, dict]:
    """
    Returns:
      chosen_action (float),
      debug dict with risks + chosen index
    """
    # Kandidaten (clipped)
    cand = np.array([a0 - delta, a0, a0 + delta], dtype=float).reshape(-1)
    cand = np.clip(cand, float(action_low[0]), float(action_high[0]))

    state = np.asarray(obs, dtype=float).reshape(-1).tolist()  # konsistent zu mlp_data_collector JSON

    risks = [mlp_model.predict_risk(risk_model, state, float(a)) for a in cand]
    risks = np.array(risks, dtype=float)

    # argmin; tie-break: bevorzugt "a0" (Index 1)
    min_val = float(np.min(risks))
    best_idxs = np.where(np.isclose(risks, min_val))[0]
    chosen_idx = int(1 if 1 in best_idxs else best_idxs[0])

    out = {
        "a_minus": float(cand[0]),
        "a_base": float(cand[1]),
        "a_plus": float(cand[2]),
        "risk_minus": float(risks[0]),
        "risk_base": float(risks[1]),
        "risk_plus": float(risks[2]),
        "chosen_idx": chosen_idx,  # 0,1,2
        "chosen_risk": float(risks[chosen_idx]),
    }
    return float(cand[chosen_idx]), out


def run_live(
    env_id: str,
    bundle_base: Path,
    mlp_path: str,
    delta: float,
    episodes: int,
    max_steps: int,
    seed: int | None,
    render: bool,
    wandb_project: str | None,
    run_name: str | None,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Load ANFIS bundle ---
    anfis_model, preprocess, y_stats, meta = load_anfis_bundle(bundle_base)
    logging.info("Loaded ANFIS bundle from %s(.model.pkl/.bundle.pkl)", bundle_base)
    logging.info("Bundle meta: %s", meta)

    # --- Load MLP risk model ---
    risk_model = mlp_model.load_risk_mlp(mlp_path)
    logging.info("Loaded MLP risk model from %s", mlp_path)

    # --- Env ---
    make_kwargs = {}
    if render:
        make_kwargs["render_mode"] = "human"

    env = gym.make(env_id, **make_kwargs)
    action_low = np.asarray(env.action_space.low, dtype=float).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=float).reshape(-1)

    # --- W&B ---
    run = None
    if wandb_project:
        cfg = {
            "env_id": env_id,
            "episodes": episodes,
            "max_steps": max_steps,
            "seed": seed,
            "bundle_base": str(bundle_base),
            "bundle_meta": meta,
            "mlp_path": mlp_path,
            "delta": float(delta),
        }
        run = init_wandb_run(project=wandb_project, job_type="counterfactual_live_run", config=cfg, run_name=run_name)

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
                a0_arr = _predict_anfis_action(anfis_model, preprocess, y_stats, obs, action_low, action_high)
                a0 = float(a0_arr.reshape(-1)[0])

                chosen_a, dbg = _choose_action_with_mlp_risk(
                    risk_model=risk_model,
                    obs=obs,
                    a0=a0,
                    delta=float(delta),
                    action_low=action_low,
                    action_high=action_high,
                )
                action = np.array([chosen_a], dtype=float)

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
                            "step/action_base_anfis": float(a0),
                            "step/action_chosen": float(chosen_a),
                            "step/risk_minus": dbg["risk_minus"],
                            "step/risk_base": dbg["risk_base"],
                            "step/risk_plus": dbg["risk_plus"],
                            "step/risk_chosen": dbg["chosen_risk"],
                            "step/chosen_idx": dbg["chosen_idx"],
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

        logging.info("Done. mean_return=%.3f ± %.3f | mean_len=%.1f | runtime=%.1fs",
                     mean_ret, std_ret, mean_len, dur)

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
    ap = argparse.ArgumentParser(description="Counterfactual Live-Run (ANFIS + MLP Risk Selection)")
    ap.add_argument("--env-id", default="InvertedPendulum-v4")
    ap.add_argument("--bundle", type=Path, required=True, help="Basispfad ohne Suffix, z.B. models/anfis_controller")
    ap.add_argument("--mlp-model", default="models/mlp_model.pth", help="Pfad zum trainierten MLP (state_dict)")
    ap.add_argument("--delta", type=float, default=0.15, help="Counterfactual Delta für a±delta")
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
        mlp_path=args.mlp_model,
        delta=args.delta,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        render=args.render,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
    )


if __name__ == "__main__":
    main()
