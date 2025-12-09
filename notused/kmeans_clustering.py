import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
 
 

def do_kmeans_clustering(file_path: str, n: int):
    # Read space/tab separated file with no header
    df = pd.read_csv(file_path, sep=r"\s+", header=None)

    # Use the first 4 columns only (ignore the 5th), ensure numeric & finite
    data = df.iloc[:, :4].apply(pd.to_numeric, errors="raise").to_numpy()
    if not np.isfinite(data).all():
        raise ValueError("NaN/inf detected in the first 4 columns.")

    kmeans = KMeans(n_clusters=n, random_state=42, n_init=20)
    labels = kmeans.fit_predict(data) 
    '''
    {
      (mean1_x, mean1_theta, mean1_xdot, mean1_thetadot): [
             [x1, theta1, xdot1, thetadot1],
             [x2, theta2, xdot2, thetadot2],
             ...
             [x2, theta2, xdot2, thetadot2],
         ],
      (mean2_x, mean2_theta, mean2_xdot, mean2_thetadot): [
             [x1, theta1, xdot1, thetadot1],
             [x2, theta2, xdot2, thetadot2],
             ...
             [x2, theta2, xdot2, thetadot2],
         ],     
        ...
    }
    '''
    clusters = {}
    for i, center in enumerate(kmeans.cluster_centers_):
        cluster_points = data[labels == i]
        clusters[tuple(center)] = cluster_points.tolist()


 
    # --- View Cluster ---
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(data)
 
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=labels, cmap="tab10", alpha=0.7)
    plt.scatter(
        pca.transform(kmeans.cluster_centers_)[:, 0],
        pca.transform(kmeans.cluster_centers_)[:, 1],
        c="red",
        marker="X",
        s=200,
        label="Centroids",
    )
    plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.title("KMeans Clustering (4D reduced to 2D with PCA)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.show()


    return clusters
 
 
if __name__ == "__main__":
    root_path = "/Users/tommykiss/mujoco-py/data" 
    input_data_path = root_path + "/AnfisTrainingSetPPO.txt"
    clusters = do_kmeans_clustering(input_data_path, n=3)
    for mean, points in clusters.items():
        print(f"Cluster mean: {mean}")
        print(f"Number of points: {len(points)}\n")