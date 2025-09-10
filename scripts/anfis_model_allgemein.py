from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import argparse
import sys
import numpy as np

import lazuardy_anfis.anfis as anfis
import lazuardy_anfis.membershipfunction as membershipfunction


# ------------------- MF-Spezifikation -------------------

@dataclass
class MFSpec:
    # Definiert die Membership Functions für jede Eingangsvariable
    mfs_per_input: list[list[list]]


# Beispiel: für jede der 4 Eingaben ein paar Gauss-MFs
DEFAULT_MF = MFSpec(
    mfs_per_input=[
        [  # x
            ["gaussmf", {"mean": 0.0, "sigma": 1.0}],
            ["gaussmf", {"mean": -2.0, "sigma": 3.0}],
            ["gaussmf", {"mean": 2.0, "sigma": 3.0}],
        ],
        [  # theta
            ["gaussmf", {"mean": 0.0, "sigma": 0.5}],
            ["gaussmf", {"mean": -1.0, "sigma": 1.0}],
            ["gaussmf", {"mean": 1.0, "sigma": 1.0}],
        ],
        [  # x_dot
            ["gaussmf", {"mean": 0.0, "sigma": 1.0}],
            ["gaussmf", {"mean": -2.0, "sigma": 3.0}],
            ["gaussmf", {"mean": 2.0, "sigma": 3.0}],
        ],
        [  # theta_dot
            ["gaussmf", {"mean": 0.0, "sigma": 0.5}],
            ["gaussmf", {"mean": -1.5, "sigma": 1.0}],
            ["gaussmf", {"mean": 1.5, "sigma": 1.0}],
        ],
    ]
)


# ------------------- Daten laden -------------------

def load_data(txt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    print(f"Lade Daten aus {txt_path}...")
    if not txt_path.exists():
        raise FileNotFoundError(f"Datei nicht gefunden: {txt_path}")

    try:
        ts = np.loadtxt(txt_path)
    except Exception as e:
        raise ValueError(f"Konnte Daten nicht laden. Urspr. Fehler: {e}")

    if ts.ndim != 2 or ts.shape[1] < 5:
        raise ValueError(f"Erwarte 2D-Array mit ≥5 Spalten (x,theta,x_dot,theta_dot,action), erhalten: {ts.shape}")

    if np.isnan(ts).any():
        raise ValueError("Daten enthalten NaNs. Bitte bereinigen.")

    X = ts[:, :4]   # 4 Eingaben
    Y = ts[:, 4]    # Aktion
    return X, Y


# ------------------- Normalisierung -------------------

def normalize(X: np.ndarray, Y: np.ndarray):
    X_mean, X_std = X.mean(axis=0), X.std(axis=0) + 1e-8
    Y_mean, Y_std = Y.mean(), Y.std() + 1e-8

    Xn = (X - X_mean) / X_std
    Yn = (Y - Y_mean) / Y_std

    stats = {"X_mean": X_mean, "X_std": X_std, "Y_mean": Y_mean, "Y_std": Y_std}
    return Xn, Yn, stats


def denormalize_y(Yn: np.ndarray, stats: dict) -> np.ndarray:
    return Yn * stats["Y_std"] + stats["Y_mean"]


# ------------------- Modellbau & Training -------------------

def build_model(X: np.ndarray, Y: np.ndarray, mf_spec: MFSpec) -> anfis.ANFIS:
    mfc = membershipfunction.MemFuncs(mf_spec.mfs_per_input)
    return anfis.ANFIS(X, Y, mfc)


def train(model: anfis.ANFIS, epochs: int = 20) -> None:
    model.trainHybridJangOffLine(epochs=epochs)


# ------------------- Auswertung & Plots -------------------

def maybe_plot(model: anfis.ANFIS, show: bool, out_dir: Path | None):
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


# ------------------- Main -------------------

def main():
    parser = argparse.ArgumentParser(description="Trainiere ein ANFIS-Modell (lazuardy_anfis) auf 4D-Pendulum-Daten.")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/AnfisTrainingSetPPO.txt"),
        help="Pfad zur Trainingsdatei (TXT, whitespace-delimited).",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Anzahl Trainings-Epochen.")
    parser.add_argument("--seed", type=int, default=42, help="Zufallsseed (Reproduzierbarkeit).")
    parser.add_argument("--no-show", action="store_true", help="Plots nicht anzeigen (nur speichern).")
    parser.add_argument(
        "--save-plots",
        type=Path,
        default=None,
        help="Wenn gesetzt: Verzeichnis, in das errors.png/results.png gespeichert werden.",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)

    # Daten laden & normalisieren
    X, Y = load_data(args.data)
    Xn, Yn, stats = normalize(X, Y)

    # Modell bauen & trainieren
    model = build_model(Xn, Yn, DEFAULT_MF)
    train(model, epochs=args.epochs)

    # Optional: ein paar Vorhersagen zurückskalieren
    preds_norm = model.fittedValues.flatten()
    preds_real = denormalize_y(preds_norm, stats)

    print("Beispiel: Erste 5 echte vs. vorhergesagte Aktionen")
    for yt, yp in zip(Y[:5], preds_real[:5]):
        print(f"true={yt:.3f}, pred={yp:.3f}")

    # nach dem Training
    Y_true = Y  # echte Aktionen (nicht normalisiert!)
    Y_pred_norm = model.fittedValues.flatten()
    Y_pred = denormalize_y(Y_pred_norm, stats)

    errors = Y_true - Y_pred
    mse = np.mean(errors**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))

    print("MSE =", mse)
    print("RMSE =", rmse)
    print("MAE =", mae)

    # Plots
    maybe_plot(model, show=not args.no_show, out_dir=args.save_plots)


if __name__ == "__main__":
    main()
