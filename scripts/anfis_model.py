from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
from sklearn.cluster import KMeans

import os
import sys

# vendor/ relativ zum Projekt-Root oder relativ zu dieser Datei:
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes from scripts/ to project root
VENDOR_DIR = os.path.join(BASE_DIR, "vendor")
sys.path.insert(0, VENDOR_DIR)

import lazuardy_anfis
import lazuardy_anfis.anfis as anfis

print("USING package:", lazuardy_anfis.__file__)
print("USING anfis.py:", anfis.__file__)


import lazuardy_anfis.membershipfunction as membershipfunction
from utils.anfis_io import make_preprocess_dict, save_anfis_bundle


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
        [["gaussmf", {"mean": 0.0, "sigma": 0.5}],
         ["gaussmf", {"mean": -1.5, "sigma": 1.0}],
         ["gaussmf", {"mean":  1.5, "sigma": 1.0}]],
    ]
)

def load_data(p: Path):
    ts = np.loadtxt(p)
    if ts.ndim != 2 or ts.shape[1] < 5:
        raise ValueError("Erwarte Spalten: x theta x_dot theta_dot action")
    X = ts[:, :4]
    y = ts[:, 4]
    return X, y

def mf_spec_from_kmeans_grid(Xn: np.ndarray, K: int = 3) -> MFSpec:
    """
    Baut pro Eingangsvariable K Gauss-MFs aus 4D-KMeans-Zentren.
    mean = Zentrumskomponente; sigma = Nachbarabstands-Heuristik.
    """
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(Xn)
    centers = kmeans.cluster_centers_  # (K, d)
    d = centers.shape[1]

    mfs_per_input: list[list[list]] = []
    for j in range(d):
        mus = np.sort(centers[:, j])        # K Mittelwerte sortieren

        # sigmas: am Rand Abstand zum einzigen Nachbarn; in der Mitte halber Nachbarabstand
        sigmas = []
        for i in range(K):
            if K == 1:
                s = 1.0
            elif i == 0:
                s = abs(mus[1] - mus[0])
            elif i == K - 1:
                s = abs(mus[-1] - mus[-2])
            else:
                s = 0.5 * abs(mus[i + 1] - mus[i - 1])
            sigmas.append(float(max(s, 1e-3)))  # numerische Untergrenze

        # in das lazuardy_anfis-Format bringen
        mfs_j = [["gaussmf", {"mean": float(mu), "sigma": float(s)}]
                 for mu, s in zip(mus, sigmas)]
        mfs_per_input.append(mfs_j)

    return MFSpec(mfs_per_input=mfs_per_input)

def train_test_split(X, y, test_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    t = int(n * (1 - test_ratio))
    train_idx, test_idx = idx[:t], idx[t:]
    return (X[train_idx], y[train_idx]), (X[test_idx], y[test_idx])

def fit_normalizer(X_train, y_train):
    X_mean, X_std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    y_mean, y_std = y_train.mean(), y_train.std() + 1e-8
    return {"X_mean": X_mean, "X_std": X_std, "y_mean": y_mean, "y_std": y_std}

def apply_normalizer_X(X, stats):
    return (X - stats["X_mean"]) / stats["X_std"]

def denorm_y(y_norm, stats):
    return y_norm * stats["y_std"] + stats["y_mean"]

def build_model(Xn, yn, mf_spec: MFSpec):
    mfc = membershipfunction.MemFuncs(mf_spec.mfs_per_input)
    return anfis.ANFIS(Xn, yn, mfc)

def metrics(y_true, y_pred):
    err = y_true - y_pred
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))  
    mae = float(np.mean(np.abs(err)))
    #ss_res = np.sum((y_true - y_pred) ** 2)               
    #ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)     
    #r2 = 1 - (ss_res / ss_tot)
    return mse, rmse, mae

def maybe_plot(model, show=True, out: Path | None = None):
    import matplotlib.pyplot as plt
    model.plotErrors()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "errors.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
    model.plotResults()
    if out:
        plt.savefig(out / "results.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

def mf_spec_from_json(json_path: Path) -> MFSpec:
    """
    Erwartet JSON aus do_kmeans_clustering_for_anfis(..., export_json_path=...):
    {
      "scaler": {"mean":[...], "scale":[...]},
      "rules": [{"centers":[...], "sigmas":[...]}, ...],  # len == K
      "meta": {"K":int, "use_cols":int, ...}
    }
    Baut pro Eingangsvariable die K Gauß-MFs aus centers/sigmas im *skalierten* Raum.
    """
    import json
    with open(json_path, "r") as f:
        cfg = json.load(f)

    rules = cfg["rules"]
    K = len(rules)
    centers = np.array([r["centers"] for r in rules], dtype=float)  # (K, d)
    sigmas  = np.array([r["sigmas"]  for r in rules], dtype=float)  # (K, d)
    K_, d = centers.shape
    assert K_ == K and sigmas.shape == (K, d), "JSON centers/sigmas Dimension mismatch"

    # pro Feature j: K MFs (mean = centers[k,j], sigma = sigmas[k,j])
    mfs_per_input: list[list[list]] = []
    for j in range(d):
        mfs_j = []
        for k in range(K):
            mu = float(centers[k, j])
            s  = float(max(sigmas[k, j], 1e-6))
            mfs_j.append(["gaussmf", {"mean": mu, "sigma": s}])
        # Optional: nach Mittelwert sortieren (stabilere Regelreihenfolge/Plots)
        mfs_j.sort(key=lambda it: it[1]["mean"])
        mfs_per_input.append(mfs_j)

    return MFSpec(mfs_per_input=mfs_per_input)

def apply_json_scaler_X(X: np.ndarray, scaler_dict: dict) -> np.ndarray:
    mean = np.array(scaler_dict["mean"], dtype=float)
    scale = np.array(scaler_dict["scale"], dtype=float)
    if X.shape[1] != mean.shape[0]:
        raise ValueError(f"Scaler dims mismatch: X has {X.shape[1]} cols, scaler has {mean.shape[0]}")
    return (X - mean) / (scale + 1e-12)


def main():
    print("started")  #debug
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/AnfisTrainingSetPPO.txt"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save-plots", type=Path, default=None)

    # NEW: KMeans-JSON (überschreibt die interne KMeans-MF-Generierung)
    ap.add_argument("--kmeans-json", type=Path, default=None,
                    help="Pfad zur JSON aus kmeans_clustering_v3.py (enthält scaler, centers, sigmas)")
    ap.add_argument("--bundle-out", type=Path, default="models/anfis_controller", help="Basispfad für Anfis-Bundle, z.B. model/anfis_controller")
    args = ap.parse_args()

    np.random.seed(args.seed)
    print("Lade Daten...") #debug

    # 1) Daten
    X, y = load_data(args.data)
    (Xtr, ytr), (Xte, yte) = train_test_split(X, y, test_ratio=0.2, seed=args.seed)

    print("X shape:", X.shape)
    print("Y shape:", y.shape)
    print("Y min/max/mean/std:", y.min(), y.max(), y.mean(), y.std())


    # 2) Normalisierung
    # 2a) Wenn JSON gegeben: X mit JSON-Scaler normalisieren (Konsistenz zu den MF-Parametern!)
    if args.kmeans_json is not None:
        import json
        with open(args.kmeans_json, "r") as f:
            j = json.load(f)
        scaler = j["scaler"]
        # nur X normalisieren; y-Stats separat (für Rückskalieren der Targets)
        Xtr_n = apply_json_scaler_X(Xtr, scaler)
        Xte_n = apply_json_scaler_X(Xte, scaler)
        y_stats = {"y_mean": ytr.mean(), "y_std": ytr.std() + 1e-8}
        ytr_n = (ytr - y_stats["y_mean"]) / y_stats["y_std"]

        # 3) MFs direkt aus JSON (bereits im skalierten Raum)
        MF_FROM_JSON = mf_spec_from_json(args.kmeans_json)
        mf_spec = MF_FROM_JSON

        # --- Debug: Membership Functions inspizieren ---
        for j, mfs in enumerate(mf_spec.mfs_per_input[:2]):  # nur die ersten 2 Inputs
            print(f"Input {j}:")
            for name, prm in mfs:
                print("  ", prm)
# ------------------------------------------------

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

    # y in Spaltenform brauchst du für lazuardy_anfis nicht explizit,
    # das Modell hat X und y schon im Konstruktor bekommen.
    # ytr_n_col = ytr_n.reshape(-1, 1)

    if hasattr(model, "trainHybridJangOffLine"):
        # Klassischer ANFIS-Train (Hybrid Jang) – nutzt LS für Consequents + GD für MFs
        print(f"Starte ANFIS-Training mit {args.epochs} Epochen...")
        model.trainHybridJangOffLine(epochs=args.epochs)

    elif hasattr(model, "fit"):
        # Fallback, falls die lazuardy-Version eine .fit-API besitzt
        print(f"Starte ANFIS-Training (fit) mit {args.epochs} Epochen...")
        model.fit(epochs=args.epochs)

    else:
        raise RuntimeError("ANFIS: Keine Trainingsmethode gefunden (trainHybridJangOffLine/fit).")


    # (Optional) Training: falls euer lazuardy_anfis.ANFIS Fit/Train-Methoden hat:
    # model.fit(epochs=args.epochs)  # <- nur falls unterstützt

    # 5) Vorhersagen & Metriken
    yhat_tr_n = model.predict(Xtr_n).reshape(-1)
    yhat_tr = denorm_y(yhat_tr_n, y_stats)
    mse_tr, rmse_tr, mae_tr = metrics(ytr, yhat_tr)

    print("Vorhersagen auf Testdaten...") #debug

    yhat_te_n = model.predict(Xte_n).reshape(-1)
    yhat_te = denorm_y(yhat_te_n, y_stats)
    mse_te, rmse_te, mae_te = metrics(yte, yhat_te)

    print("\n=== Ergebnisse ===")
    print(f"Train: MSE={mse_tr:.4f}  RMSE={rmse_tr:.4f}  MAE={mae_tr:.4f}")
    print(f"Test : MSE={mse_te:.4f}  RMSE={rmse_te:.4f}  MAE={mae_te:.4f}\n")

    maybe_plot(model, show=not args.no_show, out=args.save_plots)

        # --- NEU: ANFIS-Bundle speichern --------------------------------------
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


if __name__ == "__main__":
    main()