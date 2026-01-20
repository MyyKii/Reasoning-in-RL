from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time
import numpy as np
from sklearn.cluster import KMeans
import os
import sys

# ------------------------------ Optional W&B ------------------------------
try:
    import wandb  # type: ignore
except Exception:
    wandb = None  # type: ignore

# ------------------------------ Vendor path ------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # scripts/ -> project root
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

from lazuardy_anfis.anfis import ANFIS  # type: ignore
from lazuardy_anfis.membershipfunction import MemFuncs  # type: ignore


try:
    from utils.anfis_io import make_preprocess_dict, save_anfis_bundle
except ModuleNotFoundError:
    from anfis_io import make_preprocess_dict, save_anfis_bundle


# ------------------------------ Normalizer ------------------------------

def fit_normalizer(X: np.ndarray, y: np.ndarray) -> dict:
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-12
    y_mean = float(y.mean())
    y_std = float(y.std() + 1e-12)
    return {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}

def apply_normalizer_X(X: np.ndarray, stats: dict) -> np.ndarray:
    return (X - stats["X_mean"]) / stats["X_std"]


# ------------------------------ Data I/O ------------------------------

def load_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ts = np.loadtxt(path)
    if ts.ndim == 1 and ts.size == 5:
        ts = ts.reshape(1, 5)
    X = ts[:, :4]
    y = ts[:, 4]
    return X, y

def train_test_split(X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    rng.shuffle(idx)
    n_te = int(round(test_ratio * len(idx)))
    te_idx = idx[:n_te]
    tr_idx = idx[n_te:]
    return (X[tr_idx], y[tr_idx]), (X[te_idx], y[te_idx])


# ------------------------------ Membership Functions ------------------------------

@dataclass
class MFSpec:
    mfs_per_input: list[list[list]]

def mf_spec_from_kmeans_grid(Xn: np.ndarray, K: int = 3) -> MFSpec:
    """
    KMeans auf normalisierten Inputs -> pro Input-Dimension K Gauss-MFs
    mean = Clustercenter pro Dim
    sigma = std der Punkte im Cluster (robust, verhindert wSum=0 häufiger)
    """
    d = Xn.shape[1]
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(Xn)
    centers = kmeans.cluster_centers_  # (K, d)

    labels = kmeans.labels_
    sigmas = np.zeros_like(centers)
    for k in range(K):
        pts = Xn[labels == k]
        if pts.shape[0] < 2:
            sigmas[k, :] = 1.0
        else:
            sigmas[k, :] = np.std(pts, axis=0) + 1e-6

    mfs_per_input: list[list[list]] = []
    for j in range(d):
        mfs_j = []
        for k in range(K):
            mu = float(centers[k, j])
            s = float(max(sigmas[k, j], 1e-3))
            mfs_j.append(["gaussmf", {"mean": mu, "sigma": s}])
        mfs_j.sort(key=lambda it: it[1]["mean"])
        mfs_per_input.append(mfs_j)

    return MFSpec(mfs_per_input=mfs_per_input)


# ------------------------------ JSON KMeans Support ------------------------------

def load_kmeans_json(path: Path) -> dict:
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def apply_json_scaler_X(X: np.ndarray, scaler_dict: dict) -> np.ndarray:
    mean = np.array(scaler_dict["mean"], dtype=float)
    scale = np.array(scaler_dict["scale"], dtype=float)
    if X.shape[1] != mean.shape[0]:
        raise ValueError(f"Scaler dims mismatch: X has {X.shape[1]} cols, scaler has {mean.shape[0]}")
    return (X - mean) / (scale + 1e-12)

def mf_spec_from_json(kmeans_json_path: Path, sigma_floor: float = 0.2, sigma_scale: float = 1.5) -> MFSpec:
    """
    Unterstützt:
    A) v3-export: {"scaler":..., "rules":[{"centers":[...],"sigmas":[...]}, ...]}
    B) flach:     {"scaler":..., "centers":[...], "sigmas":[...]}
    """
    j = load_kmeans_json(kmeans_json_path)

    if "rules" in j:
        rules = j["rules"]
        centers = np.array([r["centers"] for r in rules], dtype=float)
        sigmas = np.array([r["sigmas"] for r in rules], dtype=float)
    else:
        if "centers" in j and "sigmas" in j:
            centers = np.array(j["centers"], dtype=float)
            sigmas = np.array(j["sigmas"], dtype=float)
        elif "centers_scaled" in j and "sigmas_scaled" in j:
            centers = np.array(j["centers_scaled"], dtype=float)
            sigmas = np.array(j["sigmas_scaled"], dtype=float)
        else:
            raise KeyError("KMeans-JSON: erwartete keys fehlen ('rules' oder ('centers'+'sigmas') oder ('centers_scaled'+'sigmas_scaled')).")

    if centers.shape != sigmas.shape:
        raise ValueError(f"centers shape {centers.shape} != sigmas shape {sigmas.shape}")

    K, d = centers.shape
    mfs_per_input: list[list[list]] = []
    for dim in range(d):
        mfs_j = []
        for k in range(K):
            mu = float(centers[k, dim])
            s = float(max(sigmas[k, dim], sigma_floor)) * float(sigma_scale)
            mfs_j.append(["gaussmf", {"mean": mu, "sigma": s}])
        mfs_j.sort(key=lambda it: it[1]["mean"])
        mfs_per_input.append(mfs_j)

    return MFSpec(mfs_per_input=mfs_per_input)


# ------------------------------ Model Build ------------------------------

def build_model(Xn: np.ndarray, yn: np.ndarray, mf_spec: MFSpec):
    """
    Wichtig für euren Vendor-Fork:
    - MemFuncs(...) übergeben (nicht raw list)
    - y als 1D (verhindert den backprop colY=1 out-of-bounds Pfad)
    - consequents später als (n_params, 1) initialisieren, damit predict -> (N,1) bleibt
    """
    mfc = MemFuncs(mf_spec.mfs_per_input)
    y1 = np.asarray(yn, dtype=float).reshape(-1)  # 1D
    return ANFIS(Xn, y1, mfc)


# ------------------------------ Metrics ------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, rmse, mae

def _extract_float_curve(model, candidates: list[str]) -> list[float] | None:
    for name in candidates:
        if hasattr(model, name):
            v = getattr(model, name)
            try:
                arr = np.asarray(v, dtype=float).reshape(-1)
                if arr.size == 0:
                    return None
                return [float(x) for x in arr.tolist()]
            except Exception:
                continue
    return None

def extract_vendor_error_curve(model) -> list[float] | None:
    return _extract_float_curve(model, ["errors", "Errors", "train_errors", "trainErrors", "epoch_errors", "epochErrors"])

def extract_vendor_rmse_curve(model) -> list[float] | None:
    return _extract_float_curve(model, ["rmse", "rmses", "RMSE", "RMSEs", "train_rmse", "epoch_rmse"])


# ------------------------------ Main ------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data", type=Path, default=Path("data/AnfisTrainingSet_from_PPO.txt"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--kmeans-json", type=Path, default="data/kmeans_dagger0_100k_k4.json",
                    help="Pfad zur JSON aus kmeans_clustering.py (enthält scaler, centers, sigmas)")
    ap.add_argument("--sigma-floor", type=float, default=0.2, help="Sigma-Floor beim JSON-Import (Stabilität)")
    ap.add_argument("--sigma-scale", type=float, default=1.5, help="Sigma-Scale beim JSON-Import (breitere MFs)")

    ap.add_argument("--bundle-out", type=Path, default=Path("models/anfis_controller_0"),
                    help="Basispfad für Anfis-Bundle (ohne Suffix)")

    # interpretierbare NRMSE/Proxy-Reward
    ap.add_argument("--action-range", type=float, default=6.0,
                    help="Action high-low. Für InvertedPendulum-v4 typ. 6.0 (-3..+3).")

    # DAgger Iteration als X-Achse über Runs (optional)
    ap.add_argument("--dagger-iter", type=int, default=-1, help="DAgger iteration index (optional)")

    # W&B logging (optional)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--tags", default=None, help="comma-separated, e.g. seed0,dagger,iter0")

    args = ap.parse_args()
    np.random.seed(args.seed)

    # ---------------- W&B init ----------------
    wandb_run = None
    if args.wandb_project is not None:
        if wandb is None:
            raise RuntimeError("wandb ist nicht installiert, aber --wandb-project wurde gesetzt. Installiere: pip install wandb")

        tag_list = [t.strip() for t in (args.tags or "").split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            group=args.group,
            tags=tag_list,
            config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        )

        # Epoch-basierte Kurven: X-Achse ist train/epoch
        # Wichtig: train/epoch muss in JEDEM wandb.log() für diese Metriken enthalten sein,
        # sonst werden Kurven in der UI „zusammengestaucht“.
        wandb.define_metric("train/epoch")
        for key in [
            "train/epoch_error",
            "train/epoch_rmse_vendor",
            "train/proxy_reward_vendor",
            "train/k",
            "train/sum_t",
            "train/eta",
        ]:
            wandb.define_metric(key, step_metric="train/epoch")


        # Summary
        wandb.define_metric("summary/*")
        wandb.define_metric("dataset/*")

    # ---------------- Data ----------------
    print("Lade Daten...")
    X, y = load_data(args.data)
    (Xtr, ytr), (Xte, yte) = train_test_split(X, y, test_ratio=0.2, seed=args.seed)

    if wandb_run is not None:
        # Dataset-Metadaten bewusst auf step=0 loggen, damit spätere epoch-Logs
        # (step=1..epochs) garantiert monoton bleiben.
        wandb.log({
            "dataset/n_samples": int(X.shape[0]),
            "dataset/n_train": int(Xtr.shape[0]),
            "dataset/n_test": int(Xte.shape[0]),
            "dataset/y_mean": float(np.mean(y)),
            "dataset/y_std": float(np.std(y)),
            "dataset/y_min": float(np.min(y)),
            "dataset/y_max": float(np.max(y)),
        }, step=0)

    # ---------------- Normalize + MF init ----------------
    stats = None
    scaler = None

    if args.kmeans_json is not None:
        j = load_kmeans_json(args.kmeans_json)
        scaler = j["scaler"]

        Xtr_n = apply_json_scaler_X(Xtr, scaler)
        Xte_n = apply_json_scaler_X(Xte, scaler)

        y_mean = float(np.mean(ytr))
        y_std = float(np.std(ytr) + 1e-12)
        ytr_n = (ytr - y_mean) / y_std
        y_stats = {"y_mean": y_mean, "y_std": y_std}

        mf_spec = mf_spec_from_json(args.kmeans_json, sigma_floor=args.sigma_floor, sigma_scale=args.sigma_scale)
    else:
        stats = fit_normalizer(Xtr, ytr)
        Xtr_n = apply_normalizer_X(Xtr, stats)
        Xte_n = apply_normalizer_X(Xte, stats)

        ytr_n = (ytr - stats["y_mean"]) / stats["y_std"]
        y_stats = {"y_mean": stats["y_mean"], "y_std": stats["y_std"]}

        mf_spec = mf_spec_from_kmeans_grid(Xtr_n, K=3)

    # ---------------- Model ----------------
    model = build_model(Xtr_n, ytr_n, mf_spec)
    print("Modell erstellt.")

    # Consequents explizit als 2D setzen, damit predict() -> (N,1) bleibt (Vendor erwartet oft [:,0])
    n_rules = int(np.prod([len(mfs) for mfs in mf_spec.mfs_per_input]))
    n_params = (Xtr_n.shape[1] + 1) * n_rules
    try:
        model.consequents = np.zeros((n_params, 1), dtype=float)
    except Exception:
        pass  # falls der Fork das intern anders handhabt

    # ---------------- Train (ein Call; Vendor ist nicht zuverlässig epoch-inkrementell) ----------------
    if not hasattr(model, "trainHybridJangOffLine") and not hasattr(model, "fit"):
        raise RuntimeError("ANFIS: Keine Trainingsmethode gefunden (trainHybridJangOffLine/fit).")

    print(f"Starte ANFIS-Training mit {args.epochs} Epochen...")
    train_t0 = time.time()

    import io
    import re
    import contextlib

    # Flags für den Fallback-Logger weiter unten
    logged_err_from_stdout = False
    logged_rmse_from_stdout = False
    last_epoch_logged = 0

    if hasattr(model, "trainHybridJangOffLine"):
        # Vendor gibt pro Epoche "current error" und "rmse" auf stdout aus.
        # Wir fangen das ab und loggen es danach pro Epoche nach W&B.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            model.trainHybridJangOffLine(epochs=args.epochs)

        train_out = buf.getvalue()
        print(train_out, end="")  # weiterhin in der Konsole sichtbar

        # Parse vendor prints (pro Epoche)
        errs = [float(x) for x in re.findall(r"current error:\s*([0-9eE\+\-\.]+)", train_out)]
        rmses = [float(x) for x in re.findall(r"rmse:\s*([0-9eE\+\-\.]+)", train_out)]
        ks = [float(x) for x in re.findall(r"k:\s*([0-9eE\+\-\.]+)", train_out)]
        sum_ts = [float(x) for x in re.findall(r"sum_t:\s*([0-9eE\+\-\.]+)", train_out)]
        etas = [float(x) for x in re.findall(r"eta:\s*([0-9eE\+\-\.]+)", train_out)]

        if wandb_run is not None:
            m = min(len(errs), len(rmses), int(args.epochs))
            logged_err_from_stdout = m > 0
            logged_rmse_from_stdout = m > 0
            last_epoch_logged = int(m)

            for ep in range(1, m + 1):
                r = rmses[ep - 1]
                nrmse = float(r) / float(args.action_range)  # bei euch typ. 6.0 (-3..3)

                log_row = {
                    "train/epoch": int(ep),
                    "train/epoch_error": float(errs[ep - 1]),
                    "train/epoch_rmse_vendor": float(r),
                    "train/proxy_reward_vendor": -nrmse,
                }

                # Optional: zusätzliche Metriken, falls im stdout vorhanden
                if len(ks) >= ep:
                    log_row["train/k"] = float(ks[ep - 1])
                if len(sum_ts) >= ep:
                    log_row["train/sum_t"] = float(sum_ts[ep - 1])
                if len(etas) >= ep:
                    log_row["train/eta"] = float(etas[ep - 1])

                wandb.log(log_row, step=ep)
    else:
    # Fallback falls ein anderer Fork 'fit' anbietet (meist ohne per-epoch stdout prints)
        model.fit(epochs=args.epochs)

    train_runtime = time.time() - train_t0


    # ---------------- Per-epoch curve from vendor ----------------
    err_curve = extract_vendor_error_curve(model)
    rmse_curve = extract_vendor_rmse_curve(model)

    if wandb_run is not None:
        # WICHTIG: Nicht doppelt auf step=1..epochs loggen, sonst entsteht
        # "step less than current step" und W&B ignoriert Daten.
        # Diese Fallback-Kurven nur nutzen, wenn stdout-Parsing keine Werte lieferte.
        # Falls stdout-Parsing fehlte oder nur teilweise geloggt hat, den Rest aus den Modell-Curves ergänzen.
        start_ep = int(last_epoch_logged) + 1

        if (err_curve is not None) and (start_ep <= int(args.epochs)):
            for ep, e in enumerate(err_curve[start_ep - 1 : args.epochs], start=start_ep):
                # Nur ergänzen, falls nicht bereits aus stdout geloggt.
                if logged_err_from_stdout and ep <= int(last_epoch_logged):
                    continue
                wandb.log({"train/epoch": int(ep), "train/epoch_error": float(e)}, step=ep)

        if (rmse_curve is not None) and (start_ep <= int(args.epochs)):
            for ep, r in enumerate(rmse_curve[start_ep - 1 : args.epochs], start=start_ep):
                if logged_rmse_from_stdout and ep <= int(last_epoch_logged):
                    continue
                nrmse = float(r) / float(args.action_range)
                wandb.log(
                    {
                        "train/epoch": int(ep),
                        "train/epoch_rmse_vendor": float(r),
                        "train/proxy_reward_vendor": -nrmse,
                    },
                    step=ep,
                )

        # Summary metadata
        summary_step = int(args.dagger_iter) if args.dagger_iter >= 0 else int(args.epochs)
        wandb.log(
            {
                "summary/step": summary_step,
                "summary/train_runtime_sec": float(train_runtime),
                "summary/epochs": int(args.epochs),
                "summary/dagger_iter": int(args.dagger_iter),
            }
        )

    # ---------------- Final metrics (train/test) ----------------
    yhat_tr_n = np.asarray(model.predict(Xtr_n)).reshape(-1)
    yhat_te_n = np.asarray(model.predict(Xte_n)).reshape(-1)

    yhat_tr = yhat_tr_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])
    yhat_te = yhat_te_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])

    mse_tr, rmse_tr, mae_tr = metrics(ytr, yhat_tr)
    mse_te, rmse_te, mae_te = metrics(yte, yhat_te)

    test_nrmse = rmse_te / float(args.action_range)
    test_proxy_reward = -test_nrmse

    print("\n=== Ergebnisse ===")
    print(f"Train: MSE={mse_tr:.4f}  RMSE={rmse_tr:.4f}  MAE={mae_tr:.4f}")
    print(f"Test : MSE={mse_te:.4f}  RMSE={rmse_te:.4f}  MAE={mae_te:.4f}")
    print(f"Test : NRMSE={test_nrmse:.4f}  ProxyReward={test_proxy_reward:.4f}\n")

    if wandb_run is not None:
        summary_step = int(args.dagger_iter) if args.dagger_iter >= 0 else int(args.epochs)
        wandb.log(
            {
                "summary/step": summary_step,
                "summary/train_mse": float(mse_tr),
                "summary/train_rmse": float(rmse_tr),
                "summary/train_mae": float(mae_tr),
                "summary/test_mse": float(mse_te),
                "summary/test_rmse": float(rmse_te),
                "summary/test_mae": float(mae_te),
                "summary/test_nrmse": float(test_nrmse),
                "summary/test_proxy_reward": float(test_proxy_reward),
            }
        )

    # ---------------- Save bundle ----------------
    if args.bundle_out is not None:
        if args.kmeans_json is not None:
            preprocess = make_preprocess_dict("json", scaler)  # type: ignore[arg-type]
        else:
            preprocess = make_preprocess_dict("stats", {"X_mean": stats["X_mean"], "X_std": stats["X_std"]})  # type: ignore[index]

        meta = {
            "data_path": str(args.data),
            "kmeans_json": str(args.kmeans_json) if args.kmeans_json is not None else None,
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "action_range": float(args.action_range),
            "dagger_iter": int(args.dagger_iter),
            "sigma_floor": float(args.sigma_floor),
            "sigma_scale": float(args.sigma_scale),
        }

        save_anfis_bundle(args.bundle_out, model, preprocess, y_stats, meta)
        print(f'ANFIS-Bundle gespeichert unter: "{args.bundle_out}.model.pkl" und "{args.bundle_out}.bundle.pkl"')

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
