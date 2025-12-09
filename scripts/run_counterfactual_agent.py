"""
End-to-end Training-Pipeline für:
- PPO-Teacher
- ANFIS (inkl. KMeans-MFs, Bundle speichern)
- Risk-MLP

Voraussetzung: alle anderen Skripte liegen im gleichen Ordner:
- ppo.py
- anfis_data_collector_v2.py
- kmeans_clustering_v3.py
- anfis_model_v3.py
- mlp_data_collector.py
- mlp_model.py
"""

import logging
import subprocess
import sys
from pathlib import Path

from kmeans_clustering_v3 import do_kmeans_clustering_for_anfis

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

PPO_MODEL_PATH = ROOT / "ppo_invertedpendulum.zip"
ANFIS_DATA_PATH = DATA_DIR / "AnfisTrainingSetPPO.txt"
KMEANS_JSON_PATH = DATA_DIR / "kmeans_v3.json"
ANFIS_BUNDLE_BASE = MODELS_DIR / "anfis_controller"
MLP_DATA_PATH = ROOT / "collected_data.json"
MLP_MODEL_PATH = ROOT / "mlp_model.pth"


def run_ppo_teacher():
    if PPO_MODEL_PATH.exists():
        logging.info("PPO-Modell existiert bereits (%s), überspringe Training.", PPO_MODEL_PATH)
        return
    logging.info("Starte PPO-Training...")
    subprocess.run(
        [sys.executable, str(ROOT / "ppo.py")],
        check=True,
    )
    logging.info("PPO-Training fertig.")


def run_anfis_data_collection(steps: int = 5000):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if ANFIS_DATA_PATH.exists():
        logging.info("ANFIS-Trainingsdaten existieren bereits (%s), überspringe Collection.", ANFIS_DATA_PATH)
        return
    logging.info("Starte ANFIS-Data-Collection mit PPO-Teacher...")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "anfis_data_collector_v2.py"),
            "--env_id", "InvertedPendulum-v4",
            "--model_path", str(PPO_MODEL_PATH),
            "--steps", str(steps),
            "--out_path", str(ANFIS_DATA_PATH),
            "--deterministic",
        ],
        check=True,
    )
    logging.info("ANFIS-Trainingsdaten gespeichert unter %s", ANFIS_DATA_PATH)


def run_kmeans_for_anfis(n_clusters: int = 12):
    if KMEANS_JSON_PATH.exists():
        logging.info("KMeans-JSON existiert bereits (%s), überspringe Clustering.", KMEANS_JSON_PATH)
        return
    logging.info("Starte KMeans-Clustering für ANFIS (n=%d)...", n_clusters)
    res = do_kmeans_clustering_for_anfis(
        file_path=str(ANFIS_DATA_PATH),
        n=n_clusters,
        use_cols=4,
        sigma_method="featurewise_nn",
        beta=0.55,
        alpha=1.2,
        plot=False,
        export_json_path=str(KMEANS_JSON_PATH),
    )
    logging.info("KMeans abgeschlossen. JSON gespeichert unter %s", KMEANS_JSON_PATH)
    logging.info("Cluster-Sizes: %s", res.get("cluster_sizes", "n/a"))


def run_anfis_training(epochs: int = 30):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if (ANFIS_BUNDLE_BASE.with_suffix(".model.pkl")).exists():
        logging.info("ANFIS-Bundle existiert bereits (%s.*.pkl), überspringe Training.", ANFIS_BUNDLE_BASE)
        return
    logging.info("Starte ANFIS-Training mit KMeans-JSON...")
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "anfis_model_v3.py"),
            "--data", str(ANFIS_DATA_PATH),
            "--kmeans-json", str(KMEANS_JSON_PATH),
            "--epochs", str(epochs),
            "--bundle-out", str(ANFIS_BUNDLE_BASE),
        ],
        check=True,
    )
    logging.info("ANFIS-Training + Bundle-Speicherung abgeschlossen.")


def run_mlp_data_collection():
    # mlp_data_collector.py schreibt selbst nach collected_data.json (DATA_PATH dort)
    if MLP_DATA_PATH.exists():
        logging.info("MLP-Daten existieren bereits (%s), überspringe Collection.", MLP_DATA_PATH)
        return
    logging.info("Starte MLP-Data-Collection...")
    subprocess.run(
        [sys.executable, str(ROOT / "mlp_data_collector.py")],
        check=True,
    )
    logging.info("MLP-Daten gespeichert unter %s", MLP_DATA_PATH)


def run_mlp_training():
    if MLP_MODEL_PATH.exists():
        logging.info("MLP-Modell existiert bereits (%s), überspringe Training.", MLP_MODEL_PATH)
        return
    logging.info("Starte MLP-Training...")
    subprocess.run(
        [sys.executable, str(ROOT / "mlp_model.py")],
        check=True,
    )
    logging.info("MLP-Training abgeschlossen.")


def main():
    logging.info("=== Starte komplette Training-Pipeline ===")

    run_ppo_teacher()
    run_anfis_data_collection(steps=5000)
    run_kmeans_for_anfis(n_clusters=12)
    run_anfis_training(epochs=30)
    run_mlp_data_collection()
    run_mlp_training()

    logging.info("=== Training-Pipeline vollständig abgeschlossen ===")


if __name__ == "__main__":
    main()
