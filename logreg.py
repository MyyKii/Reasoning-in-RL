import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def compute_features(state, action):
    """
    state: array-like with [x, x_dot, theta, theta_dot]
    action: float
    gibt zurück: np.array([f1, f2, f3, f4], dtype=float)
    """
    #x, x_dot, theta, theta_dot = state
    x, theta, x_dot, theta_dot = state

    f1 = max(0, abs(x) - 0.01) / 0.01 + 0.01
    f2 = max(0, abs(theta) - 0.05) / 0.05 + 0.01
    f3 = max(0, abs(x_dot)- 0.05) / 0.05 + 0.01
    f4 = max(0, abs(theta_dot) - 1.0) / 1.0 + 0.01
    f5 = max(0, abs(action) - 0.9) / 0.9 + 0.01
    return np.array([f1, f2, f3, f4, f5], dtype=float)

# 2) Daten vorbereiten
import json

# 1. Daten laden
with open("collected_data.json", "r") as f:
    raw_data = json.load(f)

#debugging
print(f"Geladene Einträge: {len(raw_data)}")
print("Erster Eintrag:", raw_data[0])


# 2. Daten in Trainingsformat bringen
X_raw = [ (d["state"], d["action"]) for d in raw_data ]
y_raw = [ d["label"] for d in raw_data ]


# Feature-Matrix bauen
X = []
for (state, action), label in zip(X_raw, y_raw):
    phi = compute_features(state, action)
    X.append(phi)
X = np.vstack(X)        # Form: (N_samples, 4)
y = np.array(y_raw)     # Form: (N_samples,)

# 3) Train/Test-Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4) Modell erstellen und trainieren
model = LogisticRegression()
model.fit(X_train, y_train)

# 5) Evaluation
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Test-Accuracy: {acc*100:.2f}%")

weights = model.coef_[0]  # model.coef_ ist ein 2D-Array: (1, n_features)
intercept = model.intercept_[0]

print("Weights:")
for i, w in enumerate(weights, 1):
    print(f"w{i}: {w:.4f}")

print(f"Bias (intercept b): {intercept:.4f}")
import numpy as np

# Label-Verteilung
print("P(label=1):", np.mean(y))

# Grobe Feature-Statistik
print("Feature-Mittelwerte:", X.mean(axis=0))
print("Feature-Std:", X.std(axis=0))


thetas = [s[1] for s, _ in X_raw]
count_over = sum(abs(t) > 0.2 for t in thetas)
total = len(thetas)

print(f"|theta| > 0.2 in {count_over} von {total} Samples "
      f"({count_over/total:.2%})")



#test
#state_new = [0.05, 0.3, 0.1, 0.5]  # x, x_dot, theta, theta_dot
#action_new = 0.95
#phi_new = compute_features(state_new, action_new).reshape(1, -1)
#risk_prob = model.predict_proba(phi_new)[0, 1]
#print(f"Risiko-Wahrscheinlichkeit: {risk_prob:.2f}")
