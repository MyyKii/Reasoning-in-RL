from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, Any
import numpy as np


# ------- Preprocessing-Container -----------------------------------------
def make_preprocess_dict(mode: str, scaler_or_stats: Dict[str, Any]) -> Dict[str, Any]:
    """
    Verpackt alle Infos zum X-Preprocessing in eine einheitliche Struktur.

    mode = 'json':
        payload = {'mean': [...], 'scale': [...]}
        -> kommt aus kmeans_clustering_v3.py (JSON-Datei)

    mode = 'stats':
        payload = {'X_mean': ndarray, 'X_std': ndarray}
        -> kommt z.B. aus fit_normalizer in anfis_model_v3.py
    """
    if mode not in ("json", "stats"):
        raise ValueError("mode must be 'json' or 'stats'")
    return {"mode": mode, "payload": scaler_or_stats}


def transform_X_with(preprocess: Dict[str, Any], X: np.ndarray) -> np.ndarray:
    """
    Wendet das passende Preprocessing auf X an.

    - Für mode='json':  (X - mean) / scale
    - Für mode='stats': (X - X_mean) / X_std
    """
    mode = preprocess["mode"]
    payload = preprocess["payload"]
    X = np.asarray(X, dtype=float)

    if mode == "json":
        mean = np.asarray(payload["mean"], dtype=float)
        scale = np.asarray(payload["scale"], dtype=float)
        return (X - mean) / (scale + 1e-12)

    elif mode == "stats":
        mean = np.asarray(payload["X_mean"], dtype=float)
        std = np.asarray(payload["X_std"], dtype=float)
        return (X - mean) / (std + 1e-12)

    else:
        raise ValueError(f"Unknown preprocess mode: {mode}")


# ------- Speichern / Laden -----------------------------------------------
def save_anfis_bundle(
    path: Path,
    model: Any,
    preprocess: Dict[str, Any],
    y_stats: Dict[str, float],
    meta: Dict[str, Any] | None = None,
) -> None:
    """
    Speichert:
      - das ANFIS-Modell   -> <path>.model.pkl
      - Zusatzinfos        -> <path>.bundle.pkl
        (preprocess, y_stats, meta)
    """
    path = Path(path)

    # 1) Modell separat speichern (kann groß sein)
    with open(path.with_suffix(".model.pkl"), "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 2) Alles andere zusammen als kleines Bundle
    payload = {
        "preprocess": preprocess,
        "y_stats": y_stats,
        "meta": meta or {},
    }
    with open(path.with_suffix(".bundle.pkl"), "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_anfis_bundle(path: Path):
    """
    Lädt das Bundle wieder.

    Rückgabe:
        model, preprocess, y_stats, meta
    """
    path = Path(path)
    with open(path.with_suffix(".model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(path.with_suffix(".bundle.pkl"), "rb") as f:
        b = pickle.load(f)
    return model, b["preprocess"], b["y_stats"], b.get("meta", {})
