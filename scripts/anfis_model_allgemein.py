from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
import numpy as np

import lazuardy_anfis.anfis as anfis
import lazuardy_anfis.membershipfunction as membershipfunction


@dataclass
class MFSpec:
    # Einfache Gauss-MF-Konfiguration je Eingangsvariable
    # Passe das bei Bedarf an (z.B. gbellmf/sigmf oder andere Parameter).
    mfs_per_input: list[list[list]]


DEFAULT_MF = MFSpec(
    mfs_per_input=[
        [
            ["gaussmf", {"mean": 0.0, "sigma": 1.0}],
            ["gaussmf", {"mean": -1.0, "sigma": 2.0}],
            ["gaussmf", {"mean": -4.0, "sigma": 10.0}],
            ["gaussmf", {"mean": -7.0, "sigma": 7.0}],
        ],
        [
            ["gaussmf", {"mean": 1.0, "sigma": 2.0}],
            ["gaussmf", {"mean": 2.0, "sigma": 3.0}],
            ["gaussmf", {"mean": -2.0, "sigma": 10.0}],
            ["gaussmf", {"mean": -10.5, "sigma": 5.0}],
        ],
    ]
)


def load_data(txt_path: Path, usecols=(1, 2, 3)) -> tuple[np.ndarray, np.ndarray]:
    print(f"Lade Daten aus {txt_path}...")
    if not txt_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {txt_path}")

    try:
        ts = np.loadtxt(txt_path, usecols=usecols)
    except Exception as e:
        raise ValueError(
            f"Konnte Daten nicht laden. Prüfe Delimiter (Standard: whitespace) "
            f"und dass die Datei mind. {len(usecols)} Spalten hat. Urspr. Fehler: {e}"
        )

    if ts.ndim != 2 or ts.shape[1] < 3:
        raise ValueError(f"Erwarte 2D-Array mit ≥3 Spalten, erhalten: {ts.shape}")

    if np.isnan(ts).any():
        raise ValueError("Daten enthalten NaNs. Bitte bereinigen.")

    X = ts[:, 0:2]
    Y = ts[:, 2]
    return X, Y


def build_model(X: np.ndarray, Y: np.ndarray, mf_spec: MFSpec) -> anfis.ANFIS:
    mfc = membershipfunction.MemFuncs(mf_spec.mfs_per_input)
    return anfis.ANFIS(X, Y, mfc)


def train(model: anfis.ANFIS, epochs: int = 20) -> None:
    model.trainHybridJangOffLine(epochs=epochs)


def evaluate(model: anfis.ANFIS, idx_check: int = 9) -> dict:
    # Kleine Metriken/Checks – erweiterbar
    results = {
        "consequents_last": float(round(model.consequents[-1][0], 6)),
        "consequents_prev": float(round(model.consequents[-2][0], 6)),
        "fitted_idx_check": float(round(model.fittedValues[idx_check][0], 6)),
    }
    return results


def ensure_expected_values(res: dict, tol: float = 0.0) -> bool:
    # Übernehme deine Referenzwerte – tol=0 entspricht exaktem Vergleich
    expected = {
        "consequents_last": -5.275538,
        "consequents_prev": -1.990703,
        "fitted_idx_check": 0.002249,
    }
    # Vergleich mit Toleranz (falls NumPy/Lib-Versionen zu minimalen Abweichungen führen)
    def close(a, b): return abs(a - b) <= tol

    return all(close(res[k], expected[k]) for k in expected)


def maybe_plot(model: anfis.ANFIS, show: bool, out_dir: Path | None):
    # lazuardy_anfis zeichnet direkt auf die aktuelle Matplotlib-Figure.
    # Wir fangen das ab: entweder anzeigen oder als Datei speichern.
    import matplotlib.pyplot as plt

    print("Plotting errors")
    model.plotErrors()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "errors.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()

    print("Plotting results")
    model.plotResults()
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "results.png", dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Trainiere ein ANFIS-Modell (lazuardy_anfis).")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("AnfisTrainingSet.txt"),
        help="Pfad zur Trainingsdatei (TXT, whitespace-delimited). Standard: ./trainingSet.txt",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Anzahl Trainings-Epochen.")
    parser.add_argument("--seed", type=int, default=42, help="Zufallsseed (Reproduzierbarkeit).")
    parser.add_argument("--no-show", action="store_true", help="Plots nicht anzeigen (nur evtl. speichern).")
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        help="Wenn gesetzt: Verzeichnis, in das errors.png/results.png gespeichert werden.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Erwarte exakt deine Referenzzahlen (Exit-Code 1 bei Abweichung).",
    )
    args = parser.parse_args()

    # Seed (für Reproduzierbarkeit – soweit von Lib unterstützt)
    np.random.seed(args.seed)

    # Daten laden
    X, Y = load_data(args.data)

    # Modell bauen & trainieren
    model = build_model(X, Y, DEFAULT_MF)
    train(model, epochs=args.epochs)

    # Auswertung + kurzer Report
    res = evaluate(model)
    print(
        f"consequents[-1][0]={res['consequents_last']}, "
        f"consequents[-2][0]={res['consequents_prev']}, "
        f"fittedValues[9][0]={res['fitted_idx_check']}"
    )

    ok = ensure_expected_values(res, tol=0.0 if args.strict else 1e-6)
    if ok:
        print("test is good")
    else:
        print("⚠️  Werte weichen von den Referenzen ab.")
        if args.strict:
            sys.exit(1)

    # Plots
    maybe_plot(model, show=not args.no_show, out_dir=args.save_plots)


if __name__ == "__main__":
    main()
