import numpy as np
from pathlib import Path

from anfis_io import (
    make_preprocess_dict,
    transform_X_with,
    save_anfis_bundle,
    load_anfis_bundle,
)


def test_preprocessing():
    print("=== Test 1: Preprocessing ===")
    # Beispiel-Daten (2 Samples, 2 Features)
    X = np.array([[1.0, 2.0],
                  [3.0, 4.0]])

    # ---------- Variante A: 'stats' (X_mean / X_std) ----------
    stats = {
        "X_mean": X.mean(axis=0),
        "X_std": X.std(axis=0) + 1e-8,  # +eps gegen Division durch 0
    }
    prep_stats = make_preprocess_dict("stats", stats)
    Xn_stats = transform_X_with(prep_stats, X)

    print("stats: normalisierte Daten:")
    print(Xn_stats)
    print("stats: Mittelwert pro Feature:", Xn_stats.mean(axis=0))
    print("stats: Std-Abweichung pro Feature:", Xn_stats.std(axis=0))
    print()

    # ---------- Variante B: 'json' (mean / scale) ----------
    json_scaler = {
        "mean": [1.0, 2.0],
        "scale": [2.0, 2.0],  # sprich: (X - mean) / 2
    }
    prep_json = make_preprocess_dict("json", json_scaler)
    Xn_json = transform_X_with(prep_json, X)

    print("json: normalisierte Daten:")
    print(Xn_json)
    # Erwartung:
    #   für X=[[1,2],[3,4]] und mean=[1,2], scale=[2,2]:
    #   -> [[0,0], [1,1]]
    print()


def test_save_and_load_bundle():
    print("=== Test 2: save_anfis_bundle / load_anfis_bundle ===")

    # Dummy-"Modell": später ist das dein echtes ANFIS-Objekt
    dummy_model = {"type": "dummy_anfis", "version": 1}

    # irgendein Preprocessing-Objekt
    preprocess = make_preprocess_dict(
        "json",
        {"mean": [0.0, 0.0], "scale": [1.0, 1.0]},
    )

    # Beispiel y-Statistik
    y_stats = {"y_mean": 0.5, "y_std": 0.1}

    # Meta-Info (beliebig)
    meta = {"note": "test-bundle"}

    # Basis-Pfad für Speicherung
    base = Path("tmp/anfis_test")
    base.parent.mkdir(parents=True, exist_ok=True)

    # Speichern
    save_anfis_bundle(base, dummy_model, preprocess, y_stats, meta)
    print(f"Bundle gespeichert unter: {base}.model.pkl / {base}.bundle.pkl")

    # Laden
    model2, preprocess2, y_stats2, meta2 = load_anfis_bundle(base)

    print("Geladenes Modell:      ", model2)
    print("Geladenes Preprocess:  ", preprocess2)
    print("Geladene y_stats:      ", y_stats2)
    print("Geladene Meta-Daten:   ", meta2)


if __name__ == "__main__":
    test_preprocessing()
    print()
    test_save_and_load_bundle()
