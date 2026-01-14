"""for s in 0 1 2 3 4; do
  python scripts/ppo_training_logging.py \
    --env-id InvertedPendulum-v4 \
    --total-timesteps 100000 \
    --seed $s \
    --model-path models/ppo_seed${s}.zip \
    --eval-freq 10000 \
    --n-eval-episodes 10 \
    --wandb-project counterfactual-agents \
    --group ppo_training_5seeds \
    --run-name ppo_train_seed${s}
done
"""


from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run

logger = logging.getLogger(__name__)


class WandbEpisodeLogger(BaseCallback):
    """Loggt Trainings-Episoden (Return/Length), sobald Monitor-Infos verfügbar sind."""
    def __init__(self, run_active: bool):
        super().__init__()
        self.run_active = run_active

    def _on_step(self) -> bool:
        if not self.run_active:
            return True

        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                log_metrics(
                    {
                        "train/episode_return": float(ep["r"]),
                        "train/episode_length": int(ep["l"]),
                    },
                    step=int(self.num_timesteps),
                )
        return True


class WandbEvalCallback(BaseCallback):
    """Evaluiert periodisch und loggt mean/std Return – das ist eure 'paper-like' Kurve."""
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        run_active: bool,
        deterministic: bool = True,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.run_active = run_active
        self.deterministic = deterministic

    def _on_step(self) -> bool:
        if self.eval_freq <= 0:
            return True

        if int(self.num_timesteps) % self.eval_freq == 0:
            rewards, lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )
            mean_r = float(np.mean(rewards))
            std_r = float(np.std(rewards))
            mean_l = float(np.mean(lengths))

            logger.info("Eval @%d steps: mean_return=%.3f ± %.3f", int(self.num_timesteps), mean_r, std_r)

            if self.run_active:
                log_metrics(
                    {
                        "eval/mean_return": mean_r,
                        "eval/std_return": std_r,
                        "eval/mean_length": mean_l,
                    },
                    step=int(self.num_timesteps),
                )
        return True


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-id", default="InvertedPendulum-v4")
    ap.add_argument("--total-timesteps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model-path", default="models/ppo_invertedpendulum.zip")

    ap.add_argument("--eval-freq", type=int, default=10_000)
    ap.add_argument("--n-eval-episodes", type=int, default=10)

    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--group", default=None)

    return ap.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    run_active = False
    if args.wandb_project:
        cfg = {
            "algo": "PPO",
            "env_id": args.env_id,
            "total_timesteps": int(args.total_timesteps),
            "seed": int(args.seed),
            "eval_freq": int(args.eval_freq),
            "n_eval_episodes": int(args.n_eval_episodes),
        }
        init_wandb_run(
            project=args.wandb_project,
            job_type="ppo_training",
            config=cfg,
            run_name=args.run_name,
            group=args.group,
            tags=["ppo", "training"],
        )
        run_active = True

    try:
        # Train env (VecMonitor for episode-Infos)
        train_env = make_vec_env(args.env_id, n_envs=1, seed=int(args.seed))
        train_env = VecMonitor(train_env)

        # Eval env (separate Seed, for stable Evaluation)
        eval_env = make_vec_env(args.env_id, n_envs=1, seed=int(args.seed) + 10_000)
        eval_env = VecMonitor(eval_env)

        model = PPO("MlpPolicy", train_env, verbose=1, seed=int(args.seed))

        cb_episode = WandbEpisodeLogger(run_active=run_active)
        cb_eval = WandbEvalCallback(
            eval_env=eval_env,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            run_active=run_active,
            deterministic=True,
        )

        logger.info("Start PPO training: timesteps=%d seed=%d", int(args.total_timesteps), int(args.seed))
        model.learn(total_timesteps=int(args.total_timesteps), callback=[cb_episode, cb_eval])

        Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.model_path)
        logger.info("Saved model to %s", args.model_path)

        train_env.close()
        eval_env.close()

    finally:
        if run_active:
            finish_wandb_run()


if __name__ == "__main__":
    main()
