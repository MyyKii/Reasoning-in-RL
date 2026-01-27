# Neuro-Fuzzy Causal Reasoning for Interpretable RL (MuJoCo InvertedPendulum-v4)

Code companion for the bachelor thesis **“Reasoning in Rule-Based Reinforcement Learning Systems using Neuro Fuzzy Causality”** (Tommy Kiss, 27.01.2026).

This repository implements a **Neuro‑Fuzzy Causal Reasoning (NFCR)** pipeline that combines:

- a **PPO teacher** (Stable‑Baselines3) for expert demonstrations and a performance baseline,
- an **ANFIS student** (Takagi–Sugeno–Kang rule base) trained offline to imitate the teacher,
- **data‑driven membership function initialization** via **k‑means** in standardized state space,
- a lightweight **DAgger‑style dataset aggregation** step to reduce covariate shift,
- an **MLP risk model** trained to predict termination likelihood from state–action pairs,
- a **counterfactual action-selection layer** that evaluates small action perturbations and executes the minimum‑risk option at inference time.

The main environment is **`InvertedPendulum-v4`** (Gymnasium MuJoCo). Observations follow:
`[x, theta, x_dot, theta_dot]` and actions are 1D continuous with typical bounds **[-3, 3]** (range = 6).  
(These conventions are also assumed by the data collectors and the proxy reward calculation.)

---

## Repository layout (core scripts)

> The scripts are referenced as `python scripts/<name>.py`. If your repo does not use a `scripts/` folder, simply drop the prefix.

### PPO (teacher + baseline)
- `train_ppo.py` – minimal PPO training (no CLI; saves to `models/ppo_invertedpendulum.zip`)
- `ppo_training_logging.py` – PPO training with periodic evaluation and W&B logging (multi‑seed friendly)
- `ppo_with_logging.py` – PPO train+eval or eval‑only; logs episode returns like the ANFIS live‑run
- `ppo_proper_logging.py` – PPO with optional **teacher‑NRMSE** logging on a fixed offline dataset (for overlays with ANFIS curves)

### ANFIS (student + deployment)
- `anfis_model_fixed.py` – trains ANFIS offline, logs vendor curves (RMSE/proxy reward), exports an **ANFIS bundle**
- `anfis_io.py` – save/load bundle helpers (`.model.pkl` + `.bundle.pkl`) + preprocessing transform
- `anfis_data_collector.py` – collects (state, teacher_action) datasets; supports DAgger‑Lite by running ANFIS while still labeling with PPO
- `anfis_live_run.py` – deploys a trained ANFIS bundle in the environment (with optional W&B logging)

### Clustering / MF visualization
- `kmeans_clustering.py` – k‑means in standardized state space + export of Gaussian MF parameters to JSON
- `mf_plot.py` – plots Gaussian membership curves from a k‑means JSON (path currently hardcoded)

### Risk model + counterfactual layer
- `mlp_data_collector.py` – collects random‑policy transitions and labels termination risk (paths/hparams currently hardcoded)
- `mlp_model.py` – trains the MLP risk predictor; provides `load_risk_mlp()` + `predict_risk()`
- `counterfactual_live_run_improved.py` – **ANFIS base action** + **(a−δ, a, a+δ)** candidate evaluation with the MLP; execute min‑risk action
- `counterfactual_live_run.py` – legacy variant (kept for reference)

### Utilities
- `wandb_utils.py` – tiny wrapper around W&B init/log/finish
- `logging_config.py` – basic console+file logger setup

---

## Installation

### Python + packages
A typical setup (Python **3.10+**) is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# core runtime
pip install numpy scipy pandas matplotlib scikit-learn torch

# RL + MuJoCo env
pip install "gymnasium[mujoco]" stable-baselines3

# optional experiment tracking
pip install wandb
```

### ANFIS vendor library
`anfis_model_fixed.py` expects `lazuardy_anfis` to be importable. The scripts automatically add `vendor/` to `PYTHONPATH` if present.

Expected structure:
```
vendor/
  lazuardy_anfis/
    anfis.py
    membershipfunction.py
    ...
```

---

## Data formats

### 1) ANFIS imitation dataset (`.txt`)
All collectors and trainers use whitespace‑separated rows:

```
x  theta  x_dot  theta_dot  action_label
```

- The first 4 values are the inputs `X`.
- `action_label` is the **teacher action** (PPO prediction), even when the behavior policy is ANFIS (DAgger‑Lite).

### 2) KMeans MF JSON (`.json`)
Created by `kmeans_clustering.py` (export option). Contains:
- `scaler`: mean/scale for standardization,
- `rules`: list of `{centers: [...], sigmas: [...]}` for Gaussian antecedents,
- `meta`: sigma method and hyperparameters.

### 3) ANFIS bundle (`<base>.model.pkl` + `<base>.bundle.pkl`)
Created by `anfis_model_fixed.py --bundle-out <base>`:
- `<base>.model.pkl`: the trained ANFIS model (pickle)
- `<base>.bundle.pkl`: preprocessing spec + output normalization stats (`y_mean`, `y_std`) + metadata

### 4) MLP risk dataset (`.json`)
Created by `mlp_data_collector.py` (hardcoded to `data/mlp_training_data.json`):
- list of dicts with at least `state`, `action`, `label`

---

## End-to-end workflow (recommended)

### Step 0 — create folders
```bash
mkdir -p data models
```

### Step 1 — train PPO teacher (baseline)
Minimal training (quick sanity check):
```bash
python scripts/train_ppo.py
# -> models/ppo_invertedpendulum.zip
```

Multi‑seed training with periodic evaluation + W&B:
```bash
for s in 0 1 2 3 4; do
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
```

### Step 2 — collect DAgger0 imitation data (teacher rollouts)
```bash
python scripts/anfis_data_collector.py \
  --env-id InvertedPendulum-v4 \
  --ppo-model-path models/ppo_seed0.zip \
  --behavior ppo \
  --steps 100000 \
  --out-path data/dagger0_seed0_100k.txt \
  --seed 0 \
  --deterministic
```

### Step 3 — run k-means + export MF parameters
`kmeans_clustering.py` currently has a hardcoded `__main__` section. The cleanest way is to call it as a module:

```bash
python -c "from scripts.kmeans_clustering import do_kmeans_clustering_for_anfis; \
do_kmeans_clustering_for_anfis( \
  file_path='data/dagger0_seed0_100k.txt', \
  n=4, use_cols=4, \
  sigma_method='featurewise_nn', beta=0.55, alpha=1.2, \
  plot=True, \
  export_json_path='data/kmeans_dagger0_seed0_100k_k4.json' \
)"
```

Notes:
- `n` controls the number of clusters/rules (**MF granularity**).
- The export JSON is consumed by `anfis_model_fixed.py --kmeans-json`.

### Step 4 — train ANFIS student (DAgger0)
```bash
python scripts/anfis_model_fixed.py \
  --data data/dagger0_seed0_100k.txt \
  --kmeans-json data/kmeans_dagger0_seed0_100k_k4.json \
  --epochs 5 \
  --seed 42 \
  --action-range 6.0 \
  --bundle-out models/anfis_controller_dagger0_seed0 \
  --dagger-iter 0 \
  --wandb-project counterfactual-agents \
  --run-name anfis_dagger0_seed0_k4
```

What gets logged (if W&B is enabled):
- `train/epoch_rmse_vendor`: vendor RMSE curve (teacher vs student)
- `train/proxy_reward_vendor`: `- (RMSE / action_range)` (imitation-fidelity proxy reward)
- `summary/test_nrmse`, `summary/test_proxy_reward`, etc.

### Step 5 — DAgger-Lite collection (DAgger1) + retrain ANFIS
Collect states induced by the student, **but still label with the PPO teacher**:

```bash
python scripts/anfis_data_collector.py \
  --ppo-model-path models/ppo_seed0.zip \
  --behavior anfis \
  --anfis-bundle models/anfis_controller_dagger0_seed0 \
  --steps 1000 \
  --out-path data/dagger1_seed0_1k.txt \
  --seed 0 \
  --deterministic
```

Optional exploration (often useful for aggregation):
```bash
# ANFIS + Gaussian action noise; teacher labels stay PPO
python scripts/anfis_data_collector.py \
  --ppo-model-path models/ppo_seed0.zip \
  --behavior anfis_noise \
  --anfis-bundle models/anfis_controller_dagger0_seed0 \
  --noise-std 0.05 \
  --teacher-mix 0.1 \
  --steps 2000 \
  --out-path data/dagger1_seed0_2k_noise.txt
```

Merge datasets:
```bash
cat data/dagger0_seed0_100k.txt data/dagger1_seed0_1k.txt > data/dagger0plus1_seed0_101k.txt
```

(Optional) recompute k-means on the aggregated dataset, then retrain ANFIS:
```bash
python scripts/anfis_model_fixed.py \
  --data data/dagger0plus1_seed0_101k.txt \
  --kmeans-json data/kmeans_dagger0plus1_seed0_k4.json \
  --epochs 5 \
  --seed 42 \
  --action-range 6.0 \
  --bundle-out models/anfis_controller_dagger1_seed0 \
  --dagger-iter 1 \
  --wandb-project counterfactual-agents \
  --run-name anfis_dagger1_seed0_k4
```

---

## Risk model (MLP)

### Step 6 — collect risk training data
`mlp_data_collector.py` uses constants at the top of the file (e.g., `NUM_EPISODES=10000`, `MAX_STEPS=500`) and writes to:
- `data/mlp_training_data.json`

Run:
```bash
python scripts/mlp_data_collector.py
```

### Step 7 — train the MLP
`mlp_model.py` reads `data/mlp_training_data.json` and saves:
- `models/mlp_model.pth`

Run:
```bash
python scripts/mlp_model.py
```

---

## Deployment / evaluation runs

### PPO baseline evaluation (episode curves)
```bash
python scripts/ppo_with_logging.py \
  --env-id InvertedPendulum-v4 \
  --model-path models/ppo_seed0.zip \
  --eval-only \
  --episodes 20 \
  --max-steps 5000 \
  --seed 0 \
  --deterministic \
  --wandb-project counterfactual-agents \
  --run-name ppo_eval_seed0
```

### ANFIS live run (student policy)
```bash
python scripts/anfis_live_run.py \
  --env-id InvertedPendulum-v4 \
  --bundle models/anfis_controller_dagger1_seed0 \
  --episodes 20 \
  --max-steps 5000 \
  --seed 0 \
  --wandb-project counterfactual-agents \
  --run-name anfis_live_seed0
```

### Counterfactual live run (ANFIS + MLP selection)
```bash
python scripts/counterfactual_live_run_improved.py \
  --env-id InvertedPendulum-v4 \
  --bundle models/anfis_controller_dagger1_seed0 \
  --mlp-model models/mlp_model.pth \
  --delta 0.15 \
  --episodes 1 \
  --max-steps 5000 \
  --seed 0 \
  --wandb-project counterfactual-agents \
  --wandb-group dagger_seed0 \
  --wandb-tags counterfactual,live \
  --run-name anfis_counterfactual_live
```

Selection logic:
- candidates: `(a0−δ, a0, a0+δ)` (clipped to action bounds),
- compute `risk_minus`, `risk_base`, `risk_plus`,
- execute `argmin(risk)`; on ties, **prefer `base`** (index 1).

To render:
- add `--render` to `anfis_live_run.py` or `counterfactual_live_run_improved.py`.

---

## Reproducing thesis-style metrics & overlays

### Proxy reward (imitation fidelity)
In the thesis, imitation quality is tracked as:

- `RMSE(student_action, teacher_action)` on a fixed offline dataset,
- `NRMSE_range = RMSE / (action_high - action_low)` (range-normalized),
- `proxy_reward = -NRMSE_range`.

`anfis_model_fixed.py` logs this per epoch as `train/proxy_reward_vendor`.  
`ppo_proper_logging.py` can log comparable teacher‑NRMSE for PPO (optional), which is useful for plots that overlay ANFIS training curves and PPO baselines.

---

## Troubleshooting

### MuJoCo / Gymnasium setup
If `InvertedPendulum-v4` fails to import/run, the usual root causes are:
- missing MuJoCo dependencies for `gymnasium[mujoco]`,
- incompatible Python version or platform wheels,
- running headless while trying to render.

Start by verifying:
```bash
python -c "import gymnasium as gym; env=gym.make('InvertedPendulum-v4'); env.reset(); print('ok')"
```

### Pickle loading errors (ANFIS bundle)
If bundle loading fails, ensure:
- `vendor/` (and `lazuardy_anfis`) is on `PYTHONPATH`,
- you run scripts from the repo root (so relative imports resolve consistently).

---

## Citation
If you use this code, please cite the accompanying thesis:

> Tommy Kiss. *Reasoning in Rule-Based Reinforcement Learning Systems using Neuro Fuzzy Causality*. Bachelor Thesis, 2026.
