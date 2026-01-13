from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import time
import numpy as np
from sklearn.cluster import KMeans

import os
import sys

# vendor/ relativ zum Projekt-Root oder relativ zu dieser Datei:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes from scripts/ to project root
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")
sys.path.insert(0, VENDOR_DIR)

# lazuardy_anfis import (liegt bei euch i.d.R. unter vendor/)
from lazuardy_anfis.anfis import ANFIS  # type: ignore

# Repo-Struktur tolerant halten
try:
    from utils.anfis_io import make_preprocess_dict, save_anfis_bundle
except ModuleNotFoundError:
    from anfis_io import make_preprocess_dict, save_anfis_bundle

try:
    from lazuardy_anfis.membershipfunction import MemFuncs  # type: ignore
except Exception:
    try:
        from lazuardy_anfis import membershipfunction  # type: ignore
        MemFuncs = membershipfunction.MemFuncs
    except Exception:
        MemFuncs = None


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

DEFAULT_MF = MFSpec(
    mfs_per_input=[
        [["gaussmf", {"mean": 0.0, "sigma": 1.0}],
         ["gaussmf", {"mean": -2.0, "sigma": 3.0}],
         ["gaussmf", {"mean":  2.0, "sigma": 3.0}]],
        [["gaussmf", {"mean": 0.0, "sigma": 0.5}],
         ["gaussmf", {"mean": -1.0, "sigma": 1.0}],
         ["gaussmf", {"mean":  1.0, "sigma": 1.0}]],
        [["gaussmf", {"mean": 0.0, "sigma": 1.0}],
         ["gaussmf", {"mean": -2.0, "sigma": 3.0}],
         ["gaussmf", {"mean":  2.0, "sigma": 3.0}]],
        [["gaussmf", {"mean": 0.0, "sigma": 1.0}],
         ["gaussmf", {"mean": -2.0, "sigma": 3.0}],
         ["gaussmf", {"mean":  2.0, "sigma": 3.0}]], 
    ]
)


def mf_spec_from_kmeans_grid(Xn: np.ndarray, K: int = 3) -> MFSpec:
    """
    KMeans auf normalisierten Inputs -> pro Input-Dimension K Gauss-MFs
    mean = Clustercenter pro Dim
    sigma = mittlere Distanz zum Center (heuristisch)
    """
    d = Xn.shape[1]
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(Xn)
    centers = kmeans.cluster_centers_  # (K, d)

    # sigma: pro cluster & dim: std der Punkte im cluster (fallback 1e-6)
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
            s = float(max(sigmas[k, j], 1e-6))
            mfs_j.append(["gaussmf", {"mean": mu, "sigma": s}])
        # Optional: nach Mittelwert sortieren (stabilere Regelreihenfolge/Plots)
        mfs_j.sort(key=lambda it: it[1]["mean"])
        mfs_per_input.append(mfs_j)

    return MFSpec(mfs_per_input=mfs_per_input)


# ------------------------------ JSON KMeans (v3) Support ------------------------------

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

def mf_spec_from_json(kmeans_json_path: Path) -> MFSpec:
    """
    Unterstützt zwei JSON-Formate:

    A) v3-export aus kmeans_clustering.py:
       {
         "scaler": {"mean":[...], "scale":[...]},
         "rules": [{"centers":[...], "sigmas":[...]}, ...],
         "meta": {...}
       }

    B) flaches Format:
       {
         "scaler": {"mean":[...], "scale":[...]},
         "centers": [[...], ...],
         "sigmas":  [[...], ...]
       }
    """
    j = load_kmeans_json(kmeans_json_path)

    # --- centers/sigmas aus "rules" oder direkt aus Top-Level ziehen ---
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
            raise KeyError(
                "KMeans-JSON hat weder 'rules' noch ('centers'+'sigmas') noch "
                "('centers_scaled'+'sigmas_scaled')."
            )

    if centers.shape != sigmas.shape:
        raise ValueError(f"centers shape {centers.shape} != sigmas shape {sigmas.shape}")

    K, d = centers.shape

    mfs_per_input: list[list[list]] = []
    for dim in range(d):
        mfs_j = []
        for k in range(K):
            mu = float(centers[k, dim])
            SIGMA_FLOOR = 0.2      # Startwert; wenn NaNs bleiben -> 0.5
            SIGMA_SCALE = 1.5      # macht MFs breiter

            s = float(max(sigmas[k, dim], SIGMA_FLOOR)) * SIGMA_SCALE

            mfs_j.append(["gaussmf", {"mean": mu, "sigma": s}])
        mfs_j.sort(key=lambda it: it[1]["mean"])
        mfs_per_input.append(mfs_j)

    return MFSpec(mfs_per_input=mfs_per_input)


# ------------------------------ Model Build ------------------------------


def build_model(Xn: np.ndarray, yn: np.ndarray, mf_spec: MFSpec):
    mf_rules = mf_spec.mfs_per_input
    yn = np.asarray(yn, dtype=float).reshape(-1)   # <-- WICHTIG: 1D
    mfc = MemFuncs(mf_rules)                       # <-- WICHTIG: MemFuncs wrapper
    return ANFIS(Xn, yn, mfc)


# ------------------------------ Metrics & Plot ------------------------------

def metrics(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mse = float(np.mean((y_true - y_pred) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return mse, rmse, mae


def _extract_anfis_error_curve(model):
    """Best-effort: versucht eine per-Epoch Fehlerkurve aus dem ANFIS-Objekt zu lesen.

    Häufige Implementierungen (inkl. lazuardy_anfis Forks) halten eine Liste wie `model.errors`,
    die auch von `model.plotErrors()` genutzt wird. Wir prüfen mehrere gängige Attributnamen.
    Gibt eine Liste[float] oder None zurück.
    """
    candidates = [
        "errors",
        "Errors",
        "train_errors",
        "trainErrors",
        "epoch_errors",
        "epochErrors",
    ]
    for name in candidates:
        if hasattr(model, name):
            v = getattr(model, name)
            try:
                import numpy as _np
                arr = _np.asarray(v, dtype=float).reshape(-1)
                if arr.size == 0:
                    return None
                return [float(x) for x in arr.tolist()]
            except Exception:
                continue
    return None


def maybe_plot(model, show: bool = True, out: Path | None = None):
    # model.plotMFs(), model.plotErrors() existieren i.d.R. in lazuardy_anfis
    if out is None and not show:
        return
    try:
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(model, "plotErrors"):
            model.plotErrors()
        if hasattr(model, "plotMFs"):
            model.plotMFs()
        if out is not None:
            import matplotlib.pyplot as plt
            plt.savefig(out, dpi=200, bbox_inches="tight")
        if show:
            import matplotlib.pyplot as plt
            plt.show()
    except Exception as e:
        print(f"[WARN] Plotting failed: {e}")


# ------------------------------ Main ------------------------------

def main():
    print("started")  # debug
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/AnfisTrainingSetPPO.txt"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save-plots", type=Path, default=None)

    # NEW: KMeans-JSON (überschreibt die interne KMeans-MF-Generierung)
    ap.add_argument(
        "--kmeans-json",
        type=Path,
        default=None,
        help="Pfad zur JSON aus kmeans_clustering_v3.py (enthält scaler, centers, sigmas)",
    )
    ap.add_argument(
        "--bundle-out",
        type=Path,
        default="models/anfis_controller",
        help="Basispfad für Anfis-Bundle, z.B. model/anfis_controller",
    )

    # W&B logging (optional)
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--group", default=None)
    ap.add_argument("--tags", default=None, help="comma-separated, e.g. seed0,dagger,iter0")

    args = ap.parse_args()

    # ---------------- W&B (optional) ----------------
    wandb_run = None
    wandb = None
    if getattr(args, "wandb_project", None):
        try:
            import wandb as _wandb  # type: ignore
            wandb = _wandb
        except Exception as e:
            raise RuntimeError(
                "wandb ist nicht installiert, aber --wandb-project wurde gesetzt. "
                "Installiere mit: pip install wandb"
            ) from e

        tag_list = [t.strip() for t in (getattr(args, "tags", None) or "").split(",") if t.strip()]
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=getattr(args, "run_name", None),
            group=getattr(args, "group", None),
            tags=tag_list,
            config={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        )
        # Saubere X-Achsen
        wandb.define_metric("dataset/step")
        wandb.define_metric("dataset/*", step_metric="dataset/step")
        wandb.define_metric("train/epoch")
        wandb.define_metric("train/*", step_metric="train/epoch")
        wandb.define_metric("summary/step")
        wandb.define_metric("summary/*", step_metric="summary/step")

    np.random.seed(args.seed)

    print("Lade Daten...")  # debug
    X, y = load_data(args.data)
    (Xtr, ytr), (Xte, yte) = train_test_split(X, y, test_ratio=0.2, seed=args.seed)

    if wandb_run is not None:
        wandb.log(
            {
                "dataset/step": 0,
                "dataset/n_samples": int(X.shape[0]),
                "dataset/n_train": int(Xtr.shape[0]),
                "dataset/n_test": int(Xte.shape[0]),
                "dataset/y_mean": float(np.mean(y)),
                "dataset/y_std": float(np.std(y)),
                "dataset/y_min": float(np.min(y)),
                "dataset/y_max": float(np.max(y)),
            }
        )

    # 2) Normalisierung + optional: JSON-KMeans
    if args.kmeans_json is not None:
        # 2a) JSON-KMeans übernimmt Skaler + MF-Parameter
        j = load_kmeans_json(args.kmeans_json)
        scaler = j["scaler"]

        # scale X with JSON scaler
        Xtr_n = apply_json_scaler_X(Xtr, scaler)
        Xte_n = apply_json_scaler_X(Xte, scaler)

        # y normalisieren weiterhin klassisch (Mean/Std auf Train)
        y_mean = float(np.mean(ytr))
        y_std = float(np.std(ytr) + 1e-12)
        ytr_n = (ytr - y_mean) / y_std

        y_stats = {"y_mean": y_mean, "y_std": y_std}

        # 3) MFs direkt aus JSON (bereits im skalierten Raum)
        mf_spec = mf_spec_from_json(args.kmeans_json)

        # --- Debug: Membership Functions inspizieren (optional) ---
        for j_in, mfs in enumerate(mf_spec.mfs_per_input[:2]):  # nur die ersten 2 Inputs
            print(f"Input {j_in}:")
            for name, prm in mfs:
                print("  ", prm)
        # ---------------------------------------------------------

    else:
        # 2b) Fallback: interner Normalizer+KMeans wie gehabt
        stats = fit_normalizer(Xtr, ytr)
        Xtr_n = apply_normalizer_X(Xtr, stats)
        Xte_n = apply_normalizer_X(Xte, stats)
        ytr_n = (ytr - stats["y_mean"]) / stats["y_std"]

        # 3) MFs automatisch aus KMeans (auf NORMALISIERTEN Inputs!)
        K = 3
        mf_spec = mf_spec_from_kmeans_grid(Xtr_n, K=K)

        # Für konsistente spätere Denorms:
        y_stats = {"y_mean": stats["y_mean"], "y_std": stats["y_std"]}

    # 4) Modell
    model = build_model(Xtr_n, ytr_n, mf_spec)
    print("Modell erstellt.")  # debug

    train_t0 = time.time()

    if hasattr(model, "trainHybridJangOffLine"):
        print(f"Starte ANFIS-Training mit {args.epochs} Epochen...")
        model.trainHybridJangOffLine(epochs=args.epochs)

    elif hasattr(model, "fit"):
        print(f"Starte ANFIS-Training (fit) mit {args.epochs} Epochen...")
        model.fit(epochs=args.epochs)

    else:
        raise RuntimeError("ANFIS: Keine Trainingsmethode gefunden (trainHybridJangOffLine/fit).")

    train_runtime = time.time() - train_t0
    if wandb_run is not None:
        wandb.log(
            {
                "summary/step": int(args.epochs),
                "summary/train_runtime_sec": float(train_runtime),
            }
        )

        errs = _extract_anfis_error_curve(model)
        if errs is not None:
            for ep_i, err in enumerate(errs[: int(args.epochs)], start=1):
                wandb.log({"train/epoch": int(ep_i), "train/epoch_error": float(err)})

    # 5) Vorhersagen (train/test) -> denormalisieren
    yhat_tr_n = np.asarray(model.predict(Xtr_n)).reshape(-1)
    yhat_te_n = np.asarray(model.predict(Xte_n)).reshape(-1)

    yhat_tr = yhat_tr_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])
    yhat_te = yhat_te_n * float(y_stats["y_std"]) + float(y_stats["y_mean"])

    mse_tr, rmse_tr, mae_tr = metrics(ytr, yhat_tr)
    mse_te, rmse_te, mae_te = metrics(yte, yhat_te)

    print("\n=== Ergebnisse ===")
    print(f"Train: MSE={mse_tr:.4f}  RMSE={rmse_tr:.4f}  MAE={mae_tr:.4f}")
    print(f"Test : MSE={mse_te:.4f}  RMSE={rmse_te:.4f}  MAE={mae_te:.4f}\n")

    if wandb_run is not None:
        wandb.log(
            {
                "summary/step": int(args.epochs),
                "summary/train_mse": float(mse_tr),
                "summary/train_rmse": float(rmse_tr),
                "summary/train_mae": float(mae_tr),
                "summary/test_mse": float(mse_te),
                "summary/test_rmse": float(rmse_te),
                "summary/test_mae": float(mae_te),
            }
        )

    maybe_plot(model, show=not args.no_show, out=args.save_plots)

    # --- ANFIS-Bundle speichern --------------------------------------
    if args.bundle_out is not None:
        # Preprocessing-Info abhängig davon, ob KMeans-JSON genutzt wurde
        if args.kmeans_json is not None:
            # scaler stammt aus der JSON und hat Keys "mean", "scale"
            preprocess = make_preprocess_dict("json", scaler)
        else:
            # stats stammt aus fit_normalizer und hat X_mean/X_std
            preprocess = make_preprocess_dict(
                "stats",
                {"X_mean": stats["X_mean"], "X_std": stats["X_std"]},
            )

        meta = {
            "data_path": str(args.data),
            "kmeans_json": str(args.kmeans_json) if args.kmeans_json is not None else None,
            "epochs": int(args.epochs),
            "seed": int(args.seed),
        }

        save_anfis_bundle(args.bundle_out, model, preprocess, y_stats, meta)
        print(
            f"ANFIS-Bundle gespeichert unter: "
            f"\"{args.bundle_out}.model.pkl\" und \"{args.bundle_out}.bundle.pkl\""
        )

    if wandb_run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
