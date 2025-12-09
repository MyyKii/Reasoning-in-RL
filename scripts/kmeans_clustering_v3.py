import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import json
from typing import Literal, Dict, Any

def _sigmas_featurewise_nn(centers_scaled: np.ndarray, beta: float = 0.55) -> np.ndarray:
    """
    pro Feature: σ_{j,k} = beta * min_{l≠k} |c_{j,k} - c_{j,l}|
    centers_scaled: (K, d)
    """
    K, d = centers_scaled.shape
    sigmas = np.zeros_like(centers_scaled)
    eps = 1e-3
    for j in range(d):
        cj = centers_scaled[:, j][:, None]     # (K,1)
        dmat = np.abs(cj - cj.T)               # (K,K)
        np.fill_diagonal(dmat, np.inf)
        dmin = np.min(dmat, axis=1)            # (K,)
        sigmas[:, j] = beta * np.maximum(dmin, eps)
    return sigmas

def _sigmas_cluster_variance(Xs: np.ndarray, labels: np.ndarray, K: int, alpha: float = 1.2) -> np.ndarray:
    """
    pro Feature: σ_{j,k} = alpha * std(x_j | cluster=k)
    """
    d = Xs.shape[1]
    sigmas = np.zeros((K, d))
    glob_var = Xs.var(axis=0, ddof=1) + 1e-8
    for k in range(K):
        Xk = Xs[labels == k]
        if len(Xk) > 1:
            var = Xk.var(axis=0, ddof=1)
        else:
            var = glob_var
        sigmas[k] = alpha * np.sqrt(var + 1e-8)
    sigmas = np.maximum(sigmas, 1e-3)
    return sigmas

def do_kmeans_clustering_for_anfis(
    file_path: str,
    n: int,
    use_cols: int = 4,
    sigma_method: Literal["featurewise_nn","cluster_variance"] = "featurewise_nn",
    beta: float = 0.55,
    alpha: float = 1.2,
    plot: bool = True,
    export_json_path: str | None = None,
) -> Dict[str, Any]:
    """
    - Liest whitespace-delimited Datei ohne Header.
    - Nutzt die ersten `use_cols` Spalten als Features.
    - Skaliert Features, macht K-Means.
    - Leitet *skalierte* Zentren & Sigmas für ANFIS ab.
    - Optional: Export als JSON.
    Returns:
      {
        'scaler': {'mean': [...], 'scale': [...]},
        'centers_scaled': (K,d) array,
        'sigmas_scaled':  (K,d) array,
        'labels': (n_samples,) array,
        'centers_orig': (K,d) array,          # nur zur Ansicht/Report
        'cluster_sizes': (K,) array,
      }
    """
    # --- Read ---
    df = pd.read_csv(file_path, sep=r"\s+", header=None)
    if df.shape[1] < use_cols:
        raise ValueError(f"Expected at least {use_cols} columns, got {df.shape[1]}")

    X = df.iloc[:, :use_cols].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("NaN/inf detected in the selected feature columns.")

    # --- Scale ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --- KMeans ---
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=20)
    labels = kmeans.fit_predict(Xs)
    centers_scaled = kmeans.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)

    sil = np.nan
    if Xs.shape[0] > n and n > 1:
        sil = silhouette_score(Xs, labels)
    print(f"Inertia: {kmeans.inertia_:.4f} | Silhouette: {sil:.4f}")
    print("Cluster sizes:", np.bincount(labels))

    # --- Sigmas (scaled space) ---
    if sigma_method == "featurewise_nn":
        sigmas_scaled = _sigmas_featurewise_nn(centers_scaled, beta=beta)
    else:
        sigmas_scaled = _sigmas_cluster_variance(Xs, labels, n, alpha=alpha)

    # --- Plot (PCA auf Xs) ---
    if plot:
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(Xs)
        centers_2d = pca.transform(centers_scaled)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
        plt.scatter(centers_2d[:, 0], centers_2d[:, 1], c="red", marker="X", s=200, label="Centroids")
        handles, _ = scatter.legend_elements()
        plt.legend(handles, [str(i) for i in range(n)], title="Clusters")
        plt.title(f"KMeans (scaled; {use_cols}D → 2D via PCA) | K={n}")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.show()

    out = {
        "scaler": {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()},
        "centers_scaled": centers_scaled,
        "sigmas_scaled": sigmas_scaled,
        "labels": labels,
        "centers_orig": centers_orig,
        "cluster_sizes": np.bincount(labels),
    }

    # --- Optional: Export als JSON für anfis.py ---
    if export_json_path is not None:
        export_obj = {
            "scaler": out["scaler"],
            "rules": [
                {
                    "centers": out["centers_scaled"][k].tolist(),
                    "sigmas": out["sigmas_scaled"][k].tolist()
                }
                for k in range(n)
            ],
            "meta": {"sigma_method": sigma_method, "beta": beta, "alpha": alpha, "K": n, "use_cols": use_cols}
        }
        with open(export_json_path, "w") as f:
            json.dump(export_obj, f, indent=2)
        print(f"Exported ANFIS params → {export_json_path}")

    return out

if __name__ == "__main__":
    root_path = "/Users/tommykiss/mujoco-py/data"
    input_data_path = root_path + "/AnfisTrainingSetPPO.txt"
    res = do_kmeans_clustering_for_anfis(
        input_data_path, n=5, use_cols=4,
        sigma_method="featurewise_nn", beta=0.55,
        plot=True,
        export_json_path= "/Users/tommykiss/mujoco-py/data/kmeans_v3.json"  # z.B. root_path + "/anfis_rules_k12.json"
    )

    centers_scaled = res["centers_scaled"]   # -> in ANFIS-MFs als c_{j,k}
    sigmas_scaled  = res["sigmas_scaled"]    # -> in ANFIS-MFs als σ_{j,k}
