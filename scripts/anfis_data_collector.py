"""
Data-Collector für MuJoCo InvertedPendulum mit PPO-Teacher.

Schreibt pro Schritt eine Zeile (whitespace-getrennt):
x  theta  x_dot  theta_dot  action

- Die ersten 4 Spalten sind die Eingaben für ANFIS (in ENV-REIHENFOLGE).
- Die 5. Spalte ist das Ziel/Label (PPO-Aktion), auf die ANFIS trainiert.

Wichtige Punkte:
- Observationen werden als Kopie gehalten (kein Buffer-Alias).
- Logging erfolgt *nach* env.step(action) mit dem resultierenden next_obs.
- Done-Handling inkl. Gymnasium terminated/truncated.
- Aktionen werden auf die Action-Space-Grenzen geclippt (float64).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
import argparse
import numpy as np

# Gymnasium prefered, fallback auf gym
try:
    import gymnasium as gym
    GYMNASIUM = True
except Exception:
    import gym  # type: ignore
    GYMNASIUM = False

from stable_baselines3 import PPO


# ------------------------------- I/O ---------------------------------

def append_row_txt(path: str, row, precision: int = 6) -> None:
    """Hängt eine Zeile an die Textdatei an (whitespace-getrennt)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fmt = f"{{:.{precision}f}}"
    with open(path, "a", encoding="utf-8") as f:
        f.write(" ".join(fmt.format(float(v)) for v in row) + "\n")


# ------------------------------ Collect ------------------------------

def collect(
    env_id: str = "InvertedPendulum-v4",
    model_path: str = "models/ppo_invertedpendulum.zip",
    steps: int = 1000,
    out_path: str = "data/AnfisTrainingSetPPO.txt",
    seed: int | None = 0,
    deterministic: bool = True,
    precision: int = 6,
    stuck_warn_every: int = 50,
) -> None:
    """
    Lädt PPO-Policy, sammelt (next_obs -> action)-Paare und schreibt:
    x, theta, x_dot, theta_dot, action
    """
    env = gym.make(env_id)

    if not Path(model_path).exists():
        print(f"FEHLER: PPO-Modell nicht gefunden: {model_path}", file=sys.stderr)
        sys.exit(1)
    model = PPO.load(model_path)

    try:
        # --- Reset + OBS-KOPIE ---
        if GYMNASIUM:
            # gymnasium.reset kann seed optional nehmen
            r = env.reset(seed=seed) if (seed is not None) else env.reset()
            obs = r[0]
        else:
            # altes gym
            if seed is not None and "seed" in env.reset.__code__.co_varnames:
                obs = env.reset(seed=seed)
            else:
                obs = env.reset()
        obs = np.array(obs, dtype=np.float64, copy=True)

        # --- Diagnose: einfacher Stuck-Detektor
        prev = None
        same_cnt = 0
        
        for t in range(steps):
            # Zustand für das Log einfrieren (KOPIE!)
            obs_for_log = np.array(obs, dtype=np.float64, copy=True)

            # Aktion aus genau diesem Zustand
            action, _ = model.predict(obs_for_log, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            low, high = env.action_space.low, env.action_space.high
            action = np.clip(action, low, high)

            # -> Jetzt s_t und a_t loggen
            x, theta, x_dot, theta_dot = obs_for_log[:4]
            append_row_txt(out_path, [x, theta, x_dot, theta_dot, float(action[0])], precision=precision)

            # Schritt ausführen und auf next_obs updaten (mit Kopie)
            step_out = env.step(action)
            if GYMNASIUM and len(step_out) == 5:
                next_obs, reward, terminated, truncated, info = step_out
                done = bool(terminated or truncated)
            else:
             next_obs, reward, done, info = step_out
            obs = np.array(next_obs, dtype=np.float64, copy=True)

            if done:
                r = env.reset()
                obs = np.array(r[0] if GYMNASIUM else r, dtype=np.float64, copy=True)

    finally:
        env.close()

    # --- Minimaler Check: laden & Form prüfen
    try:
        ts = np.loadtxt(out_path)
        print(f"[OK] Gespeichert: {ts.shape[0]} Zeilen in {out_path}")
        if ts.ndim == 1 and ts.size == 5:
            ts = ts.reshape(1, 5)
        print(f"[OK] Shape geladen: {ts.shape}  (erwarte: N x 5)")
        if ts.ndim == 2 and ts.shape[0] > 0:
            print("Erste Zeile:", ts[0].tolist())
    except Exception as e:
        print(f"[HINWEIS] Konnte {out_path} nicht erneut lesen: {e}", file=sys.stderr)


# ------------------------------ CLI ----------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Data-Collector für InvertedPendulum mit PPO-Teacher")
    ap.add_argument("--env_id", default="InvertedPendulum-v4", help="Gym/Gymnasium Env-ID")
    ap.add_argument("--model_path", default="ppo_invertedpendulum.zip", help="Pfad zum PPO-Modell (.zip)")
    ap.add_argument("--steps", type=int, default=1000, help="Anzahl Schritte")
    ap.add_argument("--out_path", default="data/AnfisTrainingSetPPO.txt", help="Ausgabedatei")
    ap.add_argument("--seed", type=int, default=0, help="Seed (None für keinen)")
    ap.add_argument("--deterministic", action="store_true", help="Deterministische Policy (SB3 predict)")
    ap.add_argument("--precision", type=int, default=6, help="Dezimalstellen in Ausgabedatei")
    ap.add_argument("--no_stuck_warn", action="store_true", help="Stuck-Warnungen deaktivieren")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    collect(
        env_id=args.env_id,
        model_path=args.model_path,
        steps=args.steps,
        out_path=args.out_path,
        seed=args.seed if args.seed is not None else None,
        deterministic=args.deterministic,
        precision=args.precision,
        stuck_warn_every=0 if args.no_stuck_warn else 50,
    )


if __name__ == "__main__":
    main()
