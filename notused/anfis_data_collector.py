#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data-Collector für MuJoCo InvertedPendulum mit PPO-Teacher.

Schreibt pro Schritt eine Zeile (whitespace-getrennt):
x  theta  x_dot  theta_dot  action

- Die ersten 4 Spalten sind die Eingaben für ANFIS (in ENV-REIHENFOLGE).
- Die 5. Spalte ist das Ziel/Label (PPO-Aktion), auf die ANFIS trainiert.
"""

import os
import sys
from pathlib import Path
import numpy as np

try:
    import gymnasium as gym
except Exception:
    import gym  

from stable_baselines3 import PPO


# --------- Observation -> Features (ENV-REIHENFOLGE) ---------
# Laut Gymnasium-Doku (InvertedPendulum):
# obs = [ x, theta, x_dot, theta_dot ]  (shape = (4,))
def extract_inputs_env_order(obs: np.ndarray) -> np.ndarray:
    o = np.asarray(obs, dtype=np.float32)
    assert o.ndim == 1 and o.size >= 4, f"Unerwartete Observation-Form: {o.shape}"
    x, theta, x_dot, theta_dot = o[0], o[1], o[2], o[3]
    return np.array([x, theta, x_dot, theta_dot], dtype=np.float32)


def append_row_txt(path: str, row) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def collect(
    env_id: str = "InvertedPendulum-v4",
    model_path: str = "ppo_invertedpendulum.zip",
    steps: int = 1000,
    out_path: str = "data/AnfisTrainingSetPPO.txt",
    seed: int = 0,
    deterministic: bool = True,
) -> None:
    """
    Lädt PPO-Policy, sammelt (obs -> action) Paare und schreibt:
    x, theta, x_dot, theta_dot, action
    """

    env = gym.make(env_id)

    if not Path(model_path).exists():
        print(f"FEHLER: PPO-Modell nicht gefunden: {model_path}", file=sys.stderr)
        sys.exit(1)
    model = PPO.load(model_path)

    try:
        r = env.reset(seed=seed) if "seed" in env.reset.__code__.co_varnames else env.reset()
        obs = r[0] if isinstance(r, tuple) else r

        for _ in range(steps):
            # a) Inputs extrahieren (nur fürs Loggen/ANFIS)
            x_vec = extract_inputs_env_order(obs)  # (4,)

            # b) PPO-Action vorhersagen (PPO bekommt die ORIGINAL-obs)
            #    Achtung: Form der Action meist (1,)
            action, _ = model.predict(obs, deterministic=deterministic)
            action = np.asarray(action, dtype=np.float32)
            # sanity: falls Scalar, in (1,) verwandeln
            if action.ndim == 0:
                action = np.array([action], dtype=np.float32)

            # c) Zeile schreiben: [x, theta, x_dot, theta_dot, action]
            append_row_txt(out_path, [*x_vec, float(action[0])])

            # d) Schritt im Env
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, _, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:
                obs, _, done, _ = step_out

            if done:
                r = env.reset()
                obs = r[0] if isinstance(r, tuple) else r

    finally:
        env.close()

    # 5) Minimaler Check: laden & Form prüfen
    ts = np.loadtxt(out_path)
    print(f"[OK] Gespeichert: {ts.shape[0]} Zeilen in {out_path}")
    print(f"[OK] Shape geladen: {ts.shape}  (erwarte: N x 5)")
    if ts.ndim == 2 and ts.shape[0] > 0:
        print("Erste Zeile:", ts[0].tolist())


if __name__ == "__main__":
    collect(
        env_id="InvertedPendulum-v4",
        model_path="ppo_invertedpendulum.zip",
        steps=1000,
        out_path="data/AnfisTrainingSetPPO.txt",
        seed=0,
        deterministic=True,
    )
