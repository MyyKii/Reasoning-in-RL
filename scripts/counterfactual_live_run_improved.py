"""counterfactual_live_run.py

Live-Run: ANFIS liefert Basisaktion a0 (unclipped + clipped).
MLP bewertet Risk für (a0-delta, a0, a0+delta) (jeweils ggf. auf Action-Space geclippt).
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
  --wandb-group dagger_seed0 \
  --wandb-tags counterfactual,live \
  --run-name anfis_counterfactual_live

W&B Logging (step-level):
- action candidates: step/action_{minus,base,plus} (+ raw variants + clip flags)
- risk candidates: step/risk_{minus,base,plus}
- selection: step/chosen_idx (0=minus, 1=base, 2=plus) + step/chosen_label
- delta details: step/chosen_delta_sign (-1/0/+1), step/chosen_delta_requested, step/chosen_delta_applied
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np

try:
    import gymnasium as gym
    _GYMNASIUM = True
except Exception:  # pragma: no cover
    import gym  # type: ignore
    _GYMNASIUM = False

# Repo-Struktur tolerant halten
try:
    from utils.anfis_io import load_anfis_bundle, transform_X_with
except ModuleNotFoundError:  # pragma: no cover
    from anfis_io import load_anfis_bundle, transform_X_with

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

import mlp_model  # nutzt load_risk_mlp() und predict_risk()


_CHOICE_LABELS: Tuple[str, str, str] = ("minus", "base", "plus")


def _predict_anfis_action(
    model,
    preprocess: dict,
    y_stats: dict,
    obs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> Tuple[float, float]:
    """Returns (a0_raw, a0_clipped)."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    x = obs[:4].reshape(1, 4)
    x_n = transform_X_with(preprocess, x)

    y_n = float(np.asarray(model.predict(x_n)).reshape(-1)[0])
    a_raw = y_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])

    low = float(action_low.reshape(-1)[0])
    high = float(action_high.reshape(-1)[0])
    a_clip = float(np.clip(a_raw, low, high))
    return float(a_raw), float(a_clip)


def _choose_action_with_mlp_risk(
    risk_model,
    obs: np.ndarray,
    a0_raw: float,
    a0: float,
    delta: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> Tuple[float, Dict]:
    """Choose among (a0-delta, a0, a0+delta) using MLP risk.

    Returns:
      chosen_action (float),
      debug dict with candidates, risks, clip flags, and chosen-index details
    """
    delta = float(abs(delta))

    low = float(action_low.reshape(-1)[0])
    high = float(action_high.reshape(-1)[0])

    # Candidate actions (raw) around the clipped base action
    cand_raw = np.array([a0 - delta, a0, a0 + delta], dtype=float).reshape(-1)
    cand = np.clip(cand_raw, low, high)

    clip_flags = (cand != cand_raw).astype(int)

    # state order is kept consistent with mlp_data_collector JSON
    state = np.asarray(obs, dtype=float).reshape(-1).tolist()

    risks = np.array(
        [mlp_model.predict_risk(risk_model, state, float(a)) for a in cand],
        dtype=float,
    )

    # argmin; tie-break: bevorzugt "base" (Index 1)
    min_val = float(np.min(risks))
    best_idxs = np.where(np.isclose(risks, min_val))[0]
    chosen_idx = int(1 if 1 in best_idxs else best_idxs[0])

    chosen_a = float(cand[chosen_idx])
    chosen_label = _CHOICE_LABELS[chosen_idx]
    chosen_delta_sign = int(chosen_idx - 1)  # -1, 0, +1
    chosen_delta_requested = float(chosen_delta_sign * delta)
    chosen_delta_applied = float(chosen_a - float(cand[1]))  # vs (clipped) base candidate

    out = {
        # Base action diagnostics
        "a0_raw": float(a0_raw),
        "a0_clipped": float(a0),
        "a0_was_clipped": int(not np.isclose(float(a0_raw), float(a0))),

        # Candidates (raw + clipped)
        "a_minus_raw": float(cand_raw[0]),
        "a_base_raw": float(cand_raw[1]),
        "a_plus_raw": float(cand_raw[2]),
        "a_minus": float(cand[0]),
        "a_base": float(cand[1]),
        "a_plus": float(cand[2]),
        "clip_minus": int(clip_flags[0]),
        "clip_base": int(clip_flags[1]),
        "clip_plus": int(clip_flags[2]),

        # Risks
        "risk_minus": float(risks[0]),
        "risk_base": float(risks[1]),
        "risk_plus": float(risks[2]),

        # Choice
        "chosen_idx": chosen_idx,  # 0,1,2
        "chosen_label": chosen_label,
        "chosen_risk": float(risks[chosen_idx]),
        "chosen_action": chosen_a,

        # Delta diagnostics
        "delta": float(delta),
        "chosen_delta_sign": chosen_delta_sign,
        "chosen_delta_requested": chosen_delta_requested,
        "chosen_delta_applied": chosen_delta_applied,
        "chosen_delta_vs_a0_raw": float(chosen_a - float(a0_raw)),
        "chosen_delta_vs_a0_clipped": float(chosen_a - float(a0)),
    }
    return chosen_a, out


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
    wandb_group: str | None = None,
    wandb_tags: List[str] | None = None,
    log_every: int = 1,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if seed is not None:
        np.random.seed(int(seed))

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
            "episodes": int(episodes),
            "max_steps": int(max_steps),
            "seed": seed,
            "bundle_base": str(bundle_base),
            "bundle_meta": meta,
            "mlp_path": mlp_path,
            "delta": float(delta),
            "choice_labels": list(_CHOICE_LABELS),
            "log_every": int(log_every),
        }
        run = init_wandb_run(
            project=wandb_project,
            job_type="counterfactual_live_run",
            config=cfg,
            run_name=run_name,
            group=wandb_group,
            tags=wandb_tags,
        )

    try:
        global_step = 0
        returns: List[float] = []
        lengths: List[int] = []
        start_t = time.time()

        for ep in range(int(episodes)):
            ep_seed = None if seed is None else int(seed) + int(ep)

            if _GYMNASIUM:
                obs, _info = env.reset(seed=ep_seed)
            else:
                # legacy gym sometimes supports seeding via reset(seed=...)
                try:
                    obs = env.reset(seed=ep_seed)  # type: ignore
                except TypeError:
                    obs = env.reset()

            obs = np.asarray(obs, dtype=float)
            ep_ret = 0.0
            ep_len = 0

            choice_counts = np.zeros(3, dtype=int)

            for t in range(int(max_steps)):
                a0_raw, a0 = _predict_anfis_action(anfis_model, preprocess, y_stats, obs, action_low, action_high)

                chosen_a, dbg = _choose_action_with_mlp_risk(
                    risk_model=risk_model,
                    obs=obs,
                    a0_raw=a0_raw,
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

                choice_counts[int(dbg["chosen_idx"])] += 1

                if run is not None and (global_step % max(1, int(log_every)) == 0):
                    log_metrics(
                        {
                            "step/reward": float(reward),

                            # ANFIS base action
                            "step/action_base_anfis_raw": float(dbg["a0_raw"]),
                            "step/action_base_anfis": float(dbg["a0_clipped"]),
                            "step/action_base_was_clipped": int(dbg["a0_was_clipped"]),

                            # candidate actions (clipped)
                            "step/action_minus": float(dbg["a_minus"]),
                            "step/action_base": float(dbg["a_base"]),
                            "step/action_plus": float(dbg["a_plus"]),

                            # candidate actions (raw)
                            "step/action_minus_raw": float(dbg["a_minus_raw"]),
                            "step/action_base_raw": float(dbg["a_base_raw"]),
                            "step/action_plus_raw": float(dbg["a_plus_raw"]),
                            "step/action_clip_minus": int(dbg["clip_minus"]),
                            "step/action_clip_base": int(dbg["clip_base"]),
                            "step/action_clip_plus": int(dbg["clip_plus"]),

                            # chosen action
                            "step/action_chosen": float(dbg["chosen_action"]),
                            "step/chosen_idx": int(dbg["chosen_idx"]),
                            "step/chosen_label": str(dbg["chosen_label"]),
                            "step/chosen_is_minus": int(dbg["chosen_idx"] == 0),
                            "step/chosen_is_base": int(dbg["chosen_idx"] == 1),
                            "step/chosen_is_plus": int(dbg["chosen_idx"] == 2),

                            # risks
                            "step/risk_minus": float(dbg["risk_minus"]),
                            "step/risk_base": float(dbg["risk_base"]),
                            "step/risk_plus": float(dbg["risk_plus"]),
                            "step/risk_chosen": float(dbg["chosen_risk"]),

                            # delta diagnostics
                            "step/delta": float(dbg["delta"]),
                            "step/chosen_delta_sign": int(dbg["chosen_delta_sign"]),
                            "step/chosen_delta_requested": float(dbg["chosen_delta_requested"]),
                            "step/chosen_delta_applied": float(dbg["chosen_delta_applied"]),
                            "step/chosen_delta_vs_a0_raw": float(dbg["chosen_delta_vs_a0_raw"]),
                            "step/chosen_delta_vs_a0_clipped": float(dbg["chosen_delta_vs_a0_clipped"]),

                            "step/episode": int(ep),
                            "step/t": int(t),
                        },
                        step=global_step,
                    )

                obs = np.asarray(next_obs, dtype=float)
                if done:
                    break

            returns.append(ep_ret)
            lengths.append(ep_len)

            logging.info("Episode %d/%d | return=%.3f | len=%d | choice_counts=%s",
                         ep + 1, episodes, ep_ret, ep_len, choice_counts.tolist())

            if run is not None:
                denom = max(1, int(ep_len))
                log_metrics(
                    {
                        "episode/return": float(ep_ret),
                        "episode/length": int(ep_len),
                        "episode/index": int(ep),
                        "episode/chosen_minus_frac": float(choice_counts[0] / denom),
                        "episode/chosen_base_frac": float(choice_counts[1] / denom),
                        "episode/chosen_plus_frac": float(choice_counts[2] / denom),
                        "episode/chosen_minus_count": int(choice_counts[0]),
                        "episode/chosen_base_count": int(choice_counts[1]),
                        "episode/chosen_plus_count": int(choice_counts[2]),
                    },
                    step=global_step,
                )

        dur = time.time() - start_t
        mean_ret = float(np.mean(returns)) if returns else float("nan")
        std_ret = float(np.std(returns)) if returns else float("nan")
        mean_len = float(np.mean(lengths)) if lengths else float("nan")

        logging.info("DONE | mean_return=%.3f ± %.3f | mean_len=%.1f | runtime=%.1fs",
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

    # W&B
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb-group", default=None)
    ap.add_argument("--wandb-tags", default=None, help="Comma-separated, e.g. dagger,seed0,iter1")
    ap.add_argument("--log-every", type=int, default=1, help="Log every N environment steps (W&B).")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    tags = None
    if args.wandb_tags:
        tags = [t.strip() for t in str(args.wandb_tags).split(",") if t.strip()]

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
        wandb_group=args.wandb_group,
        wandb_tags=tags,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
