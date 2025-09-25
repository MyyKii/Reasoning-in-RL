import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def do_kmeans_clustering(file_path: str, n: int, use_cols: int = 4):
    """
    - Reads a whitespace-delimited file with no header.
    - Uses the first `use_cols` columns as features (default 4).
    - Scales features before K-Means and PCA.
    - Returns a dict: { centroid_tuple_in_original_units : [points_in_original_units] }.
    """
    # --- Read your space/tab separated file (no header) ---
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    if df.shape[1] < use_cols:
        raise ValueError(f"Expected at least {use_cols} columns, got {df.shape[1]}")

    # --- First N columns as features; enforce numeric & finite ---
    X = df.iloc[:, :use_cols].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(X).all():
        raise ValueError("NaN/inf detected in the selected feature columns.")

    # --- Scale (important for distance-based methods like K-Means) ---
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # --- K-Means (robust init) ---
    kmeans = KMeans(n_clusters=n, random_state=42, n_init=20)
    labels = kmeans.fit_predict(Xs)

    # Diagnostics
    sil = np.nan
    if Xs.shape[0] > n and n > 1:
        sil = silhouette_score(Xs, labels)
    print(f"Inertia: {kmeans.inertia_:.4f} | Silhouette: {sil:.4f}")

    # --- Build cluster dict: keys/values in ORIGINAL units for readability ---
    centers_scaled = kmeans.cluster_centers_
    centers_orig = scaler.inverse_transform(centers_scaled)

    clusters = {}
    for i, center in enumerate(centers_orig):
        pts_orig = X[labels == i]  # original units
        clusters[tuple(center)] = pts_orig.tolist()

    # Quick text summary
    print("Cluster sizes:", np.bincount(labels))
    print("Centroids (original units):")
    print(pd.DataFrame(centers_orig,
                       columns=[f"feat_{j}" for j in range(use_cols)]))

    # --- View clusters in 2D via PCA on the SCALED data ---
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(Xs)
    centers_2d = pca.transform(centers_scaled)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1],
                          c=labels, cmap="tab10", alpha=0.7)
    plt.scatter(centers_2d[:, 0], centers_2d[:, 1],
                c="red", marker="X", s=200, label="Centroids")
    handles, _ = scatter.legend_elements()
    plt.legend(handles, [str(i) for i in range(n)], title="Clusters")
    plt.title("KMeans Clustering (scaled; 4D → 2D via PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()

    return clusters


if __name__ == "__main__":
    root_path = "/Users/tommykiss/mujoco-py/data"
    input_data_path = root_path + "/AnfisTrainingSetPPO.txt"

    # If you want to use all 5 numbers per line, call with use_cols=5
    clusters = do_kmeans_clustering(input_data_path, n=3, use_cols=4)

    for mean, points in clusters.items():
        print(f"Cluster mean (orig units): {mean}")
        print(f"Number of points: {len(points)}\n")
