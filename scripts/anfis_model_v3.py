from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import lazuardy_anfis.anfis as anfis
import lazuardy_anfis.membershipfunction as membershipfunction

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
    for j in range(d):                      # für jede der d Dimensionen
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
    return mse, rmse, mae

def maybe_plot(model, show=True, out: Path | None = None):
    import matplotlib.pyplot as plt
    model.plotErrors()
    if out:
        out.mkdir(parents=True, exist_ok=True)
        plt.savefig(out / "errors.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()
    model.plotResults()
    if out:
        plt.savefig(out / "results.png", dpi=150, bbox_inches="tight")
    if show: plt.show()
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/AnfisTrainingSetPPO.txt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save-plots", type=Path, default=None)
    args = ap.parse_args()

    np.random.seed(args.seed)

    # 1) Daten
    X, y = load_data(args.data)
    (Xtr, ytr), (Xte, yte) = train_test_split(X, y, test_ratio=0.2, seed=args.seed)

    # 2) Normalisierung (auf TRAIN fitten!)
    stats = fit_normalizer(Xtr, ytr)
    Xtr_n = apply_normalizer_X(Xtr, stats)
    Xte_n = apply_normalizer_X(Xte, stats)
    # y wird im Modell normalisiert gelernt:
    ytr_n = (ytr - stats["y_mean"]) / stats["y_std"]

    # 3) MFs automatisch aus KMeans (auf NORMALISIERTEN Inputs!)
    K = 3  # starte mit 3; später 2/4 testen
    MF_FROM_KMEANS = mf_spec_from_kmeans_grid(Xtr_n, K=K)

    # 4) Modell
    model = build_model(Xtr_n, ytr_n, MF_FROM_KMEANS)


    # 5) Vorhersagen (zurückskalieren) & Metriken
    # Auf TRAIN:
    yhat_tr_n = model.predict(Xtr_n).reshape(-1)
    yhat_tr = denorm_y(yhat_tr_n, stats)
    mse_tr, rmse_tr, mae_tr = metrics(ytr, yhat_tr)

    # Auf TEST:
    yhat_te_n = model.predict(Xte_n).reshape(-1)
    yhat_te = denorm_y(yhat_te_n, stats)
    mse_te, rmse_te, mae_te = metrics(yte, yhat_te)

    print("\n=== Ergebnisse ===")
    print(f"Train: MSE={mse_tr:.4f}  RMSE={rmse_tr:.4f}  MAE={mae_tr:.4f}")
    print(f"Test : MSE={mse_te:.4f}  RMSE={rmse_te:.4f}  MAE={mae_te:.4f}\n")

    # 6) Plots (optional)
    maybe_plot(model, show=not args.no_show, out=args.save_plots)

if __name__ == "__main__":
    main()
