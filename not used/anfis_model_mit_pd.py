import os
import numpy as np


try:
    import gymnasium as gym
except Exception:
    import gym  


# Laut Gymnasium-Doku (InvertedPendulum):
# obs[0]=x, obs[1]=theta, obs[2]=x_dot, obs[3]=theta_dot
def extract_inputs(obs: np.ndarray) -> np.ndarray:
    o = np.asarray(obs, dtype=np.float32)
    assert o.ndim == 1 and o.size >= 4, f"Unerwartete Observation-Form: {o.shape}"
    x       = o[0]
    theta   = o[1]
    x_dot   = o[2]
    th_dot  = o[3]
    # Reihenfolge für Datei: x, x_dot, theta, theta_dot
    return np.array([x, x_dot, theta, th_dot], dtype=np.float32)


# TODO: change to ppo agent
def teacher_action(features: np.ndarray, kp: float = 10.0, kd: float = 2.0) -> float:
    # features: [x, x_dot, theta, theta_dot]
    theta   = float(features[2])
    th_dot  = float(features[3])
    return -kp * theta - kd * th_dot


def map_to_action_space(y: float, action_space):
    # Kontinuierlich (Box): clip auf Bounds, Form anpassen
    if hasattr(action_space, "shape"):
        lo = np.float32(action_space.low[0]) if np.ndim(action_space.low) else np.float32(action_space.low)
        hi = np.float32(action_space.high[0]) if np.ndim(action_space.high) else np.float32(action_space.high)
        u = np.clip(np.float32(y), lo, hi)
        # Form: (action_dim,) – InvertedPendulum hat 1D Aktion
        return np.array([u], dtype=np.float32)
    # Diskret (zur Not): Vorzeichen-Schwelle
    return int(y > 0.0)


def append_row_txt(path: str, row) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(" ".join(f"{v:.6f}" for v in row) + "\n")


def collect(env_id: str = "InvertedPendulum-v4",
            steps: int = 1000,
            out_path: str = "data/AnfisTrainingSet.txt",
            seed: int = 0) -> None:
    env = gym.make(env_id)
    try:
        # reset-API: Gymnasium -> (obs, info), Gym -> obs
        r = env.reset(seed=seed) if "seed" in env.reset.__code__.co_varnames else env.reset()
        obs = r[0] if isinstance(r, tuple) else r

        for _ in range(steps):
            x_vec = extract_inputs(obs)               # (4,)
            y     = teacher_action(x_vec)             # scalar target
            append_row_txt(out_path, [*x_vec, y])     # 5 Werte

            action = map_to_action_space(y, env.action_space)
            step_out = env.step(action)

            # Gymnasium: obs, reward, terminated, truncated, info
            if len(step_out) == 5:
                obs, _, terminated, truncated, _ = step_out
                done = bool(terminated or truncated)
            else:  # Gym: obs, reward, done, info
                obs, _, done, _ = step_out

            if done:
                r = env.reset()
                obs = r[0] if isinstance(r, tuple) else r
    finally:
        env.close()

    # Minimal-Check: laden & Form ausgeben
    ts = np.loadtxt(out_path)
    print(f"[OK] Gespeichert: {ts.shape[0]} Zeilen in {out_path}")
    print(f"[OK] Shape geladen: {ts.shape}  (erwarte: N x 5)")
    print("Erste Zeile:", ts[0].tolist() if ts.ndim == 2 and ts.shape[0] > 0 else "—")


if __name__ == "__main__":
    collect()
