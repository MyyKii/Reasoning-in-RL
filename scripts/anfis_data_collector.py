"""
Data-Collector für MuJoCo InvertedPendulum mit PPO-Teacher.

Schreibt pro Schritt eine Zeile (whitespace-getrennt):
x  theta  x_dot  theta_dot  action_label

- Die ersten 4 Spalten sind die Eingaben für ANFIS (in ENV-Reihenfolge).
- Die 5. Spalte ist das Ziel/Label (Teacher-Aktion, i.d.R. PPO),
  auch wenn die Umgebung ggf. von ANFIS gesteuert wird (DAgger-Lite).

Modi:
- behavior=ppo:         env.step(PPO), log(PPO)
- behavior=ppo_noise:   env.step(PPO + noise), log(PPO)
- behavior=anfis:       env.step(ANFIS), log(PPO)            <-- DAgger-Lite
- behavior=anfis_noise: env.step(ANFIS + noise), log(PPO)    <-- DAgger-Lite + Exploration

Optional:
- teacher_mix in [0,1]: mit Wahrscheinlichkeit teacher_mix wird PPO als Behavior genutzt,
  ansonsten der gewählte Behavior-Controller (ANFIS/ANFIS+noise). Label bleibt PPO.

W&B Logging (optional):
- Aktivieren mit: --wandb-project <name>
- Step-level throttled via --log-every (default 1000)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import argparse
import numpy as np

# --- ensure vendor/lazuardy_anfis is importable if present (for unpickling ANFIS model) ---
_THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = _THIS_FILE.parents[1]  # scripts/ -> project root
VENDOR_DIR = PROJECT_ROOT / "vendor"
if VENDOR_DIR.exists():
    sys.path.insert(0, str(VENDOR_DIR))

# Gymnasium preferred, fallback to gym
try:
    import gymnasium as gym
    GYMNASIUM = True
except Exception:
    import gym  
    GYMNASIUM = False

from stable_baselines3 import PPO

try:
    from utils.anfis_io import load_anfis_bundle, transform_X_with
except ModuleNotFoundError:
    from anfis_io import load_anfis_bundle, transform_X_with


# ------------------------------- I/O ---------------------------------

def append_row_txt(path: str, row, precision: int = 6) -> None:
    """Hängt eine Zeile an die Textdatei an (whitespace-getrennt)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fmt = f"{{:.{precision}f}}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(" ".join(fmt.format(float(v)) for v in row) + "\n")


# ------------------------------ Helpers ------------------------------

def predict_anfis_action(
    model,
    preprocess: dict,
    y_stats: dict,
    obs: np.ndarray,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> np.ndarray:
    """ANFIS(norm(obs)) -> y_norm -> denorm -> clip."""
    obs = np.asarray(obs, dtype=float).reshape(-1)
    x = obs[:4].reshape(1, 4)
    x_n = transform_X_with(preprocess, x)

    y_n = np.asarray(model.predict(x_n)).reshape(-1)[0]
    a = y_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])
    a_arr = np.array([a], dtype=float)
    return np.clip(a_arr, action_low, action_high)


# ------------------------------ Collect ------------------------------

def collect(
    env_id: str = "InvertedPendulum-v4",
    ppo_model_path: str = "models/ppo_seed0.zip",
    steps: int = 1000,
    out_path: str = "data/AnfisTrainingSetPPO.txt",
    seed: int | None = 0,
    deterministic: bool = True,
    precision: int = 6,
    behavior: str = "ppo",
    anfis_bundle: str | None = None,
    noise_std: float = 0.0,
    teacher_mix: float = 0.0,
    # --- W&B (optional) ---
    wandb_project: str | None = None,
    run_name: str | None = None,
    group: str | None = None,
    tags: str | None = None,
    log_every: int = 1000,
) -> None:
    """
    Sammelt (state -> teacher_action)-Paare.
    Behavior bestimmt, welche Aktion tatsächlich in env.step() geht.
    """
    if not Path(ppo_model_path).exists():
        print(f"FEHLER: PPO-Modell nicht gefunden: {ppo_model_path}", file=sys.stderr)
        sys.exit(1)

    behavior = behavior.lower().strip()
    if behavior not in {"ppo", "ppo_noise", "anfis", "anfis_noise"}:
        raise ValueError(f"Unknown behavior: {behavior}")

    teacher_mix = float(teacher_mix)
    if not (0.0 <= teacher_mix <= 1.0):
        raise ValueError("teacher_mix must be in [0,1]")

    # W&B init (optional)
    wandb_run = None
    wandb = None
    if wandb_project:
        try:
            import wandb as _wandb  # type: ignore
            wandb = _wandb
        except Exception as e:
            raise RuntimeError(
                "wandb ist nicht installiert, aber --wandb-project wurde gesetzt. "
                "Installiere mit: pip install wandb"
            ) from e

        tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]
        wandb_run = wandb.init(
            project=wandb_project,
            name=run_name,
            group=group,
            tags=tag_list,
            config={
                "env_id": env_id,
                "ppo_model_path": str(ppo_model_path),
                "steps": int(steps),
                "out_path": str(out_path),
                "seed": seed,
                "deterministic": bool(deterministic),
                "precision": int(precision),
                "behavior": behavior,
                "anfis_bundle": str(anfis_bundle) if anfis_bundle else None,
                "noise_std": float(noise_std),
                "teacher_mix": float(teacher_mix),
                "log_every": int(log_every),
            },
        )
        # Saubere X-Achse in W&B
        wandb.define_metric("collector/step")
        wandb.define_metric("collector/*", step_metric="collector/step")

    t0 = time.time()

    env = gym.make(env_id)
    action_low = np.asarray(env.action_space.low, dtype=float).reshape(-1)
    action_high = np.asarray(env.action_space.high, dtype=float).reshape(-1)

    teacher = PPO.load(ppo_model_path)

    # Optional: load ANFIS bundle if needed
    anfis_model = None
    preprocess = None
    y_stats = None
    if behavior.startswith("anfis"):
        if anfis_bundle is None:
            raise ValueError("behavior=anfis* requires --anfis-bundle")
        anfis_model, preprocess, y_stats, meta = load_anfis_bundle(Path(anfis_bundle))

    rng = np.random.default_rng(seed if seed is not None else None)

    ep_ret = 0.0
    ep_len = 0
    ep_idx = 0

    try:
        # Reset
        if GYMNASIUM:
            r = env.reset(seed=seed) if (seed is not None) else env.reset()
            obs = r[0]
        else:
            obs = env.reset()
        obs = np.array(obs, dtype=np.float64, copy=True)

        for t in range(steps):
            obs_for_log = np.array(obs, dtype=np.float64, copy=True)

            # --- Teacher label (immer PPO) ---
            a_teacher, _ = teacher.predict(obs_for_log, deterministic=deterministic)
            a_teacher = np.asarray(a_teacher, dtype=np.float64).reshape(-1)
            a_teacher = np.clip(a_teacher, action_low, action_high)

            # --- Behavior action ---
            use_teacher = (rng.random() < teacher_mix)

            if use_teacher or behavior.startswith("ppo"):
                a_env = a_teacher.copy()
            else:
                # anfis / anfis_noise
                a_env = predict_anfis_action(
                    anfis_model, preprocess, y_stats,
                    obs_for_log, action_low, action_high
                )

            # Optional: noise on behavior action (never on label)
            if behavior.endswith("noise") and noise_std > 0.0 and (not use_teacher):
                a_env = a_env + rng.normal(0.0, noise_std, size=a_env.shape)
                a_env = np.clip(a_env, action_low, action_high)

            # --- Log state + teacher label ---
            x, theta, x_dot, theta_dot = obs_for_log[:4]
            append_row_txt(
                out_path,
                [x, theta, x_dot, theta_dot, float(a_teacher[0])],
                precision=precision,
            )

            # Step
            step_out = env.step(a_env)
            if GYMNASIUM and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
                next_obs, reward, done, info = step_out

            ep_ret += float(reward)
            ep_len += 1

            # W&B step logging (throttled)
            if wandb_run is not None and (t % max(int(log_every), 1) == 0):
                wandb.log(
                    {
                        "collector/step": int(t),
                        "collector/step_reward": float(reward),
                        "collector/teacher_action": float(a_teacher[0]),
                        "collector/behavior_action": float(a_env[0]),
                        "collector/action_diff": float(a_env[0] - a_teacher[0]),
                        "collector/use_teacher": int(use_teacher),
                        "collector/episode": int(ep_idx),
                    }
                )

            obs = np.array(next_obs, dtype=np.float64, copy=True)

            if done:
                # episode summary
                if wandb_run is not None:
                    wandb.log(
                        {
                            "collector/step": int(t),
                            "collector/episode_return": float(ep_ret),
                            "collector/episode_length": int(ep_len),
                            "collector/episode": int(ep_idx),
                        }
                    )

                ep_ret = 0.0
                ep_len = 0
                ep_idx += 1

                r = env.reset()
                obs = np.array(r[0] if GYMNASIUM else r, dtype=np.float64, copy=True)

    finally:
        env.close()
        if wandb_run is not None:
            wandb.log(
                {
                    "collector/step": int(steps),
                    "collector/runtime_sec": float(time.time() - t0),
                }
            )
            wandb.finish()

    # minimaler Check
    ts = np.loadtxt(out_path)
    if ts.ndim == 1 and ts.size == 5:
        ts = ts.reshape(1, 5)
    print(f"[OK] Gespeichert: {ts.shape[0]} Zeilen in {out_path} | shape={ts.shape}")


# ------------------------------ CLI ----------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Data-Collector für InvertedPendulum mit PPO-Teacher / DAgger-Lite")
    ap.add_argument("--env-id", default="InvertedPendulum-v4")
    ap.add_argument("--ppo-model-path", default="models/ppo_invertedpendulum.zip")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--out-path", default="data/AnfisTrainingSetPPO.txt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--precision", type=int, default=6)

    ap.add_argument("--behavior", default="ppo", choices=["ppo", "ppo_noise", "anfis", "anfis_noise"])
    ap.add_argument("--anfis-bundle", default=None, help="Basispfad ohne Suffix, z.B. models/anfis_controller")
    ap.add_argument("--noise-std", type=float, default=0.0, help="Stddev Gaussian Noise auf Behavior-Action")
    ap.add_argument("--teacher-mix", type=float, default=0.0, help="Wahrscheinlichkeit PPO als Behavior zu nehmen (0..1)")

    # W&B (optional)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--tags", default=None, help="comma-separated, e.g. seed0,dagger,iter0")
    ap.add_argument("--log-every", type=int, default=1000, help="W&B logging frequency (steps)")

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        env_id=args.env_id,
        ppo_model_path=args.ppo_model_path,
        steps=args.steps,
        out_path=args.out_path,
        seed=args.seed,
        deterministic=args.deterministic,
        precision=args.precision,
        behavior=args.behavior,
        anfis_bundle=args.anfis_bundle,
        noise_std=args.noise_std,
        teacher_mix=args.teacher_mix,
        wandb_project=args.wandb_project,
        run_name=args.run_name,
        group=args.group,
        tags=args.tags,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
