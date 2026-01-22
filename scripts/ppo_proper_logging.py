"""
ppo_with_logging.py

PPO Baseline (Train optional) + Evaluation mit W&B Logging.

Erweiterung:
- Optional: loggt -NRMSE der *aktuellen* PPO-Policy gegen einen PPO-Teacher
  auf einem fixen State-Set (aus eurem DAgger0 Dataset).

W&B Metrics (neu):
- train/ppo_rmse_teacher
- train/ppo_nrmse_teacher
- train/ppo_neg_nrmse_teacher

ANFIS-kompatible Duplikate (für Overlays):
- train/epoch_rmse_vendor      == RMSE (Student vs Teacher)
- train/proxy_reward_vendor    == -NRMSE (Student vs Teacher)

Achsen:
- train/epoch       (Eval-Index 0..K)
- train/timesteps   (PPO Timesteps)
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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Teacher-NRMSE helpers (Split kompatibel zu anfis_model.py)
# ---------------------------------------------------------------------

def _load_state_action_txt(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Erwartet whitespace-getrennt: x theta x_dot theta_dot action_label."""
    ts = np.loadtxt(path)
    if ts.ndim == 1:
        ts = ts.reshape(1, -1)
    if ts.ndim != 2 or ts.shape[1] < 5:
        raise ValueError(f"Erwarte >=5 Spalten (x theta x_dot theta_dot action). Got shape={ts.shape}")
    X = np.asarray(ts[:, :4], dtype=float)
    y = np.asarray(ts[:, 4], dtype=float).reshape(-1, 1)
    return X, y


def _train_test_split_like_anfis_model(X: np.ndarray, y: np.ndarray, test_ratio: float, seed: int):
    """Gleiche Logik wie train_test_split() in anfis_model.py."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    t = int(n * (1.0 - float(test_ratio)))
    train_idx, test_idx = idx[:t], idx[t:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])


def prepare_teacher_eval_set(
    env_id: str,
    data_path: Path,
    split: str,
    test_ratio: float,
    seed: int,
    max_samples: int,
    teacher_model_path: Optional[str],
    deterministic: bool,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, str]:
    """
    Bereitet (X_eval, a_teacher_eval) für RMSE/NRMSE vor.

    split: train | test | all
    teacher_model_path:
      - wenn gesetzt: Teacher-Targets via PPO.load(...).predict(X_eval)
      - sonst: Targets aus Dataset-Spalte 5

    NRMSE-Normalisierung: RMSE / std(a_teacher) (konsistent zur y_std Idee in anfis_model.py).
    """
    split = split.lower().strip()
    if split not in {"train", "test", "all"}:
        raise ValueError("split must be one of: train, test, all")

    if not data_path.exists():
        raise FileNotFoundError(f"NRMSE dataset not found: {data_path}")

    # Action bounds aus der ENV (für konsistentes Clipping)
    env = gym.make(env_id)
    try:
        action_low = np.asarray(env.action_space.low, dtype=float).reshape(-1)
        action_high = np.asarray(env.action_space.high, dtype=float).reshape(-1)
    finally:
        env.close()

    X_all, y_labels = _load_state_action_txt(data_path)

    if split == "all":
        X_eval = X_all
        y_eval = y_labels
    else:
        (Xtr, ytr), (Xte, yte) = _train_test_split_like_anfis_model(
            X_all, y_labels, test_ratio=float(test_ratio), seed=int(seed)
        )
        X_eval, y_eval = (Xtr, ytr) if split == "train" else (Xte, yte)

    # optional subsample (fix via seed)
    if max_samples is not None and int(max_samples) > 0 and X_eval.shape[0] > int(max_samples):
        rng = np.random.default_rng(int(seed))
        sel = rng.choice(np.arange(X_eval.shape[0]), size=int(max_samples), replace=False)
        X_eval = X_eval[sel]
        y_eval = y_eval[sel]

    X_eval = np.asarray(X_eval, dtype=float)

    if teacher_model_path and Path(teacher_model_path).exists():
        teacher = PPO.load(teacher_model_path)
        a_teacher, _ = teacher.predict(X_eval, deterministic=bool(deterministic))
        a_teacher = np.asarray(a_teacher, dtype=float)
        teacher_source = f"model:{teacher_model_path}"
    else:
        a_teacher = np.asarray(y_eval, dtype=float)
        teacher_source = f"labels:{str(data_path)}"

    # Shape (N, action_dim)
    a_teacher = a_teacher.reshape(X_eval.shape[0], -1)

    # clip to env bounds
    a_teacher = np.clip(a_teacher, action_low, action_high)

    y_std = float(np.std(a_teacher)) + 1e-8
    return X_eval, a_teacher, y_std, action_low, action_high, teacher_source


class TeacherNRMSECallback(BaseCallback):
    """Loggt RMSE/NRMSE der PPO-Policy gegen Teacher-Targets auf einem fixen State-Set."""

    def __init__(
        self,
        X_eval: np.ndarray,
        a_teacher: np.ndarray,
        y_std: float,
        action_low: np.ndarray,
        action_high: np.ndarray,
        eval_freq: int,
        deterministic: bool,
        run_active: bool,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.X_eval = np.asarray(X_eval, dtype=float)
        self.a_teacher = np.asarray(a_teacher, dtype=float)
        self.y_std = float(y_std)
        self.action_low = np.asarray(action_low, dtype=float).reshape(-1)
        self.action_high = np.asarray(action_high, dtype=float).reshape(-1)
        self.eval_freq = int(eval_freq)
        self.deterministic = bool(deterministic)
        self.run_active = bool(run_active)

        self._last_eval_step: int = -10**18
        self._eval_idx: int = 0

    def _compute(self) -> Tuple[float, float, float]:
        a_pred, _ = self.model.predict(self.X_eval, deterministic=self.deterministic)
        a_pred = np.asarray(a_pred, dtype=float).reshape(self.a_teacher.shape)
        a_pred = np.clip(a_pred, self.action_low, self.action_high)

        err = a_pred - self.a_teacher
        rmse = float(np.sqrt(np.mean(err ** 2)))
        # Action-Range aus env bounds (bei InvertedPendulum typ. 6)
        action_range = float(np.mean(self.action_high - self.action_low)) + 1e-8

        # Range-NRMSE (das ist die Variante, die zu --action-range=6 passt)
        nrmse_range = float(rmse / action_range)
        neg_nrmse_range = -nrmse_range

        # Optional: Std-NRMSE nur als Debug behalten
        nrmse_std = float(rmse / self.y_std)
        neg_nrmse_std = -nrmse_std

        return rmse, nrmse_range, neg_nrmse_range, nrmse_std, neg_nrmse_std, action_range


    def _log_now(self) -> None:
        rmse, nrmse_range, neg_nrmse_range, nrmse_std, neg_nrmse_std, action_range = self._compute()
        pseudo_epoch = int(self._eval_idx) + 1  # 1..K (wie ANFIS)
        metrics = {
            "train/epoch": pseudo_epoch,
            "train/timesteps": int(self.num_timesteps),  # optional, aber praktisch
            "train/ppo_rmse_teacher": rmse,

            # PPO "official" jetzt range-basiert:
            "train/ppo_nrmse_teacher": nrmse_range,
            "train/ppo_neg_nrmse_teacher": neg_nrmse_range,
            "train/ppo_action_range": action_range,

            # Debug: std-basiert (optional)
            "train/ppo_nrmse_std_teacher": nrmse_std,
            "train/ppo_neg_nrmse_std_teacher": neg_nrmse_std,
            "train/ppo_nrmse_y_std": float(self.y_std),

            # ANFIS-Overlay Keys: unbedingt range-basiert setzen!
            "train/epoch_rmse_vendor": rmse,
            "train/proxy_reward_vendor": neg_nrmse_range,
        }


        if self.run_active:
            log_metrics(metrics, step=int(self.num_timesteps))

        self._eval_idx += 1

    def _on_training_start(self) -> None:
        self._log_now()
        self._last_eval_step = int(self.num_timesteps)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True
        if int(self.num_timesteps) - int(self._last_eval_step) >= int(self.eval_freq):
            self._log_now()
            self._last_eval_step = int(self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if int(self.num_timesteps) != int(self._last_eval_step):
            self._log_now()


# ---------------------------------------------------------------------
# Standard Eval (Return/Length)
# ---------------------------------------------------------------------

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

            if run_active:
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
            log_metrics(
                {
                    "summary/mean_return": mean_ret,
                    "summary/std_return": std_ret,
                    "summary/mean_length": mean_len,
                    "summary/runtime_sec": float(dur),
                },
                step=episodes,
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
    # Teacher-NRMSE
    teacher_model_path: Optional[str],
    nrmse_data: Optional[str],
    nrmse_eval_freq: int,
    nrmse_split: str,
    nrmse_test_ratio: float,
    nrmse_max_samples: int,
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
            # NRMSE config
            "teacher_model_path": teacher_model_path,
            "nrmse_data": nrmse_data,
            "nrmse_eval_freq": int(nrmse_eval_freq),
            "nrmse_split": str(nrmse_split),
            "nrmse_test_ratio": float(nrmse_test_ratio),
            "nrmse_max_samples": int(nrmse_max_samples),
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

        # Optional: prepare fixed eval set once
        teacher_cb = None
        teacher_source = None
        if nrmse_data:
            X_eval, a_teacher, y_std, a_low, a_high, teacher_source = prepare_teacher_eval_set(
                env_id=env_id,
                data_path=Path(nrmse_data),
                split=nrmse_split,
                test_ratio=float(nrmse_test_ratio),
                seed=int(seed or 0),
                max_samples=int(nrmse_max_samples),
                teacher_model_path=teacher_model_path,
                deterministic=deterministic,
            )
        else:
            X_eval = a_teacher = a_low = a_high = None
            y_std = None

        if eval_only:
            if not model_file.exists():
                raise FileNotFoundError(f"PPO model not found: {model_path}")
            logger.info("Loading PPO model: %s", model_path)
            model = PPO.load(model_path)

            # Eval-only: log teacher NRMSE exactly once (if configured)
            if run_active and nrmse_data:
                cb = TeacherNRMSECallback(
                    X_eval=X_eval,
                    a_teacher=a_teacher,
                    y_std=y_std,
                    action_low=a_low,
                    action_high=a_high,
                    eval_freq=0,
                    deterministic=deterministic,
                    run_active=True,
                )
                cb.model = model  # type: ignore[attr-defined]
                cb.num_timesteps = 0  # type: ignore[attr-defined]
                cb._log_now()
                if teacher_source:
                    log_metrics({"summary/ppo_teacher_source": str(teacher_source)}, step=0)

        else:
            logger.info("Training PPO: env=%s timesteps=%d seed=%s", env_id, total_timesteps, str(seed))
            env = gym.make(env_id)
            try:
                env.reset(seed=seed)
                model = PPO("MlpPolicy", env, verbose=1, seed=seed)

                if nrmse_data and int(nrmse_eval_freq) > 0:
                    teacher_cb = TeacherNRMSECallback(
                        X_eval=X_eval,
                        a_teacher=a_teacher,
                        y_std=y_std,
                        action_low=a_low,
                        action_high=a_high,
                        eval_freq=int(nrmse_eval_freq),
                        deterministic=deterministic,
                        run_active=run_active,
                    )
                    if run_active and teacher_source:
                        log_metrics({"summary/ppo_teacher_source": str(teacher_source)}, step=0)

                model.learn(total_timesteps=int(total_timesteps), callback=teacher_cb)
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
    ap = argparse.ArgumentParser(description="PPO Baseline + optional Teacher -NRMSE logging to W&B")
    ap.add_argument("--env-id", default="InvertedPendulum-v4")
    ap.add_argument("--model-path", default="models/ppo_invertedpendulum.zip")
    ap.add_argument("--total-timesteps", type=int, default=100_000)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--log-steps", action="store_true")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)

    # Teacher-NRMSE
    ap.add_argument("--teacher-model-path", default=None, help="Pfad zum PPO-Teacher (.zip).")
    ap.add_argument("--nrmse-data", default=None, help="Dataset: x theta x_dot theta_dot action_label")
    ap.add_argument("--nrmse-eval-freq", type=int, default=0, help="Eval-Frequenz in Timesteps (0=aus)")
    ap.add_argument("--nrmse-split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--nrmse-test-ratio", type=float, default=0.2)
    ap.add_argument("--nrmse-max-samples", type=int, default=5000)

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
        teacher_model_path=args.teacher_model_path,
        nrmse_data=args.nrmse_data,
        nrmse_eval_freq=args.nrmse_eval_freq,
        nrmse_split=args.nrmse_split,
        nrmse_test_ratio=args.nrmse_test_ratio,
        nrmse_max_samples=args.nrmse_max_samples,
    )


if __name__ == "__main__":
    main()
