#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, List, Tuple, Dict, Any
import time
import json
import numpy as np

# Optional: Gym/Gymnasium kompatibel nutzen
try:
    import gymnasium as gym
except Exception:
    import gym  # type: ignore


# ---------- Feature-Extractor-Beispiele ----------

def extract_inverted_pendulum(obs: np.ndarray) -> np.ndarray:
    """
    Erwartet obs in Reihenfolge: [x, x_dot, theta, theta_dot]
    Passt an DEIN Env an (ggf. Mapping notwendig).
    """
    # Falls dein Env etwas wie [cos(theta), sin(theta), theta_dot, x, x_dot] liefert,
    # musst du das hier entsprechend umordnen/rekonstruieren.
    return np.asarray([
        obs[0],   # x
        obs[1],   # x_dot
        obs[2],   # theta
        obs[3],   # theta_dot
    ], dtype=np.float32)


# Beispiel: simple "Teacher"-Policy (PD auf Winkel), nur als Platzhalter
def teacher_policy_pd_ip(states: np.ndarray,
                         kp: float = 10.0, kd: float = 2.0,
                         u_max: float = 10.0) -> float:
    """
    states: [x, x_dot, theta, theta_dot]
    Ziel: theta -> 0, theta_dot -> 0 (aufrecht)
    """
    theta = float(states[2])
    theta_dot = float(states[3])
    u = -kp * theta - kd * theta_dot
    # Clip auf Env-Grenzen
    return float(np.clip(u, -u_max, u_max))


# ---------- Collector ----------

@dataclass
class AnfisDataCollector:
    feature_fn: Callable[[np.ndarray], np.ndarray]
    target_fn: Callable[[np.ndarray], float]  # gibt Zielwert (z. B. Action) zurück
    include_timestamp: bool = False
    buffer_maxlen: Optional[int] = None  # None = unendlich
    meta: Dict[str, Any] = field(default_factory=dict)

    # intern
    _rows: List[List[float]] = field(default_factory=list, init=False)
    _episode_idx: int = field(default=0, init=False)
    _t0: float = field(default_factory=time.time, init=False)

    def start_episode(self) -> None:
        self._episode_idx += 1

    def record(self, obs: np.ndarray) -> float:
        """
        Nimmt eine Beobachtung auf, berechnet (X -> Y) und wird in buffer gespeichert.
        Gibt den target (z. B. Action) zurück – praktisch, wenn du den auch im Env benutzen willst.
        """
        x_vec = self.feature_fn(obs)            # shape (D,)
        y_val = float(self.target_fn(x_vec))    # scalar

        if x_vec.ndim != 1:
            raise ValueError(f"feature_fn muss Vektor (D,) liefern, bekam {x_vec.shape}")

        row = list(map(float, x_vec)) + [y_val]  # [x1, x2, ..., xD, y]
        if self.include_timestamp:
            row = [time.time() - self._t0, float(self._episode_idx)] + row

        # Rolling-Buffer
        if self.buffer_maxlen is not None and len(self._rows) >= self.buffer_maxlen:
            self._rows.pop(0)
        self._rows.append(row)

        return y_val

    def flush(self) -> None:
        # hier nichts nötig – Platzhalter falls du später Streaming willst
        pass

    def clear(self) -> None:
        self._rows.clear()

    def as_array(self) -> np.ndarray:
        if not self._rows:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(self._rows, dtype=np.float32)

    def save_txt(self, path: Path, header: Optional[str] = None) -> Path:
        arr = self.as_array()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(path, arr, fmt="%.6f")  # whitespace-delimited
        if header:
            (path.with_suffix(path.suffix + ".meta.json")).write_text(header, encoding="utf-8")
        return path

    def save_csv(self, path: Path, header_cols: Optional[List[str]] = None) -> Path:
        arr = self.as_array()
        path.parent.mkdir(parents=True, exist_ok=True)
        if header_cols is not None:
            np.savetxt(path, arr, delimiter=",", fmt="%.6f",
                       header=",".join(header_cols), comments="")
        else:
            np.savetxt(path, arr, delimiter=",", fmt="%.6f")
        return path

    def build_header_cols(self, D: int) -> List[str]:
        cols = [f"x{i+1}" for i in range(D)] + ["target"]
        if self.include_timestamp:
            cols = ["t", "episode"] + cols
        return cols

    def meta_json(self) -> str:
        return json.dumps(self.meta, ensure_ascii=False, indent=2)


# ---------- CLI-Demo: sammelt Daten und speichert sie ----------

def run_collect_demo(env_id: str = "CartPole-v1",
                     steps: int = 5000,
                     txt_out: Path = Path("trainingSet.txt"),
                     use_csv: bool = False) -> None:
    """
    Demo-Loop:
    - reset -> step -> record (states -> target=teacher_action)
    - Speichert whitespace-TXT (Standard) ODER CSV
    """
    env = gym.make(env_id)
    # WARNUNG: Setze diesen Extractor auf DEINE Beobachtungen um!
    feature_fn = extract_inverted_pendulum
    target_fn = teacher_policy_pd_ip

    collector = AnfisDataCollector(
        feature_fn=feature_fn,
        target_fn=target_fn,
        include_timestamp=False,
        meta={"env_id": env_id, "note": "targets are teacher actions (PD on theta)"}
    )

    obs, _ = env.reset() if hasattr(env, "reset") and "return_info" in env.reset.__code__.co_varnames else (env.reset(), {})
    collector.start_episode()

    n = 0
    done = False
    while n < steps:
        # 1) Feature -> Target (z. B. Action)
        try:
            y = collector.record(np.asarray(obs, dtype=np.float32))
        except Exception as e:
            raise RuntimeError(f"Feature/Target-Funktion passt nicht zur Obs-Form: {e}")

        # 2) In der Demo benutzen wir die gleiche Action auch für den Env-Step
        action = np.array([y], dtype=np.float32) if hasattr(env.action_space, "shape") else int(y > 0.0)
        step_out = env.step(action)
        if len(step_out) == 5:
            obs, reward, terminated, truncated, info = step_out
            done = bool(terminated or truncated)
        else:
            obs, reward, done, info = step_out

        n += 1
        if done:
            obs, _ = env.reset() if len(getattr(env.reset, "__code__", None).co_varnames) > 0 else (env.reset(), {})
            collector.start_episode()
            done = False

    arr = collector.as_array()

    # Speichern
    D = arr.shape[1] - 1  # ohne target; timestamp/episode nicht inkludiert, da disabled
    header_cols = collector.build_header_cols(D)

    if use_csv:
        collector.save_csv(Path(txt_out).with_suffix(".csv"), header_cols)
    else:
        # TXT + Meta-JSON mit Spaltennamen, damit du weißt was drin ist
        meta = {
            "columns": header_cols,
            "env_id": env_id,
            **collector.meta
        }
        collector.save_txt(Path(txt_out), header=json.dumps(meta, ensure_ascii=False))

    print(f"Saved {arr.shape[0]} samples to {txt_out.resolve() if not use_csv else Path(txt_out).with_suffix('.csv').resolve()}")
