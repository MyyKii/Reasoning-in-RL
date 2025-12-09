import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

JSON_PATH = "data/kmeans_v3.json"

with open(JSON_PATH, "r") as f:
    data = json.load(f)

rules = data.get("rules", [])
meta = data.get("meta", {})
K = meta.get("K", len(rules))
use_cols = meta.get("use_cols", None)

if not rules:
    raise ValueError("Keine 'rules' im JSON gefunden.")

centers_list = []
sigmas_list = []
for i, r in enumerate(rules):
    c = r.get("centers")
    s = r.get("sigmas")
    if c is None or s is None:
        raise ValueError(f"Rule {i} hat keine 'centers' oder 'sigmas'.")
    if len(c) != len(s):
        raise ValueError(f"Rule {i}: centers und sigmas haben unterschiedliche Längen.")
    centers_list.append(c)
    sigmas_list.append(s)

centers = np.asarray(centers_list)  
sigmas  = np.asarray(sigmas_list)   

K_detected, D = centers.shape
if use_cols is None:
    use_cols = D
elif use_cols != D:
    print(f"Warnung: meta.use_cols={use_cols}, aber Daten haben D={D}. Verwende D={D}.")
    use_cols = D

# ---------- Plot 1: mean and error bars ----------
# For every feature, plot centers with error bars (sigmas)
"""for feat in range(use_cols):
    plt.figure(figsize=(8, 4.5))
    x = np.arange(K_detected)
    y = centers[:, feat]
    yerr = sigmas[:, feat]
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=4)
    plt.title(f"Feature {feat} – Cluster-Mittelwerte mit σ")
    plt.xlabel("Cluster-Index")
    plt.ylabel("Center")
    plt.grid(True, alpha=0.3)
    out_path = Path(f"feature_{feat}_centers_sigmas.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()"""

# ---------- Plot 2: Fuzzy-Membership-Curves ----------
# For each feature, plot K Gaussian curves (centers/sigmas)
def gaussian(x, c, s):
    s = np.maximum(s, 1e-12)
    return np.exp(-0.5 * ((x - c) / s)**2)

for feat in range(use_cols):
    c_feat = centers[:, feat]
    s_feat = sigmas[:, feat]
    # useful x-range
    x_min = np.min(c_feat - 4*s_feat)
    x_max = np.max(c_feat + 4*s_feat)
    # if sigmas are zero or inf/nan, use default range
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = -2.0, 2.0
    xs = np.linspace(x_min, x_max, 600)

    plt.figure(figsize=(8, 4.5))
    for k in range(K_detected):
        ys = gaussian(xs, c_feat[k], s_feat[k])
        # plot every curve seperatly for legend
        plt.plot(xs, ys, label=f"C{k}: c={c_feat[k]:.3f}, σ={s_feat[k]:.3f}")
    plt.title(f"Feature {feat} – Fuzzy Membership Functions (Gauss)")
    plt.xlabel("x")
    plt.ylabel("μ(x)")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper right", ncol=1, fontsize=8)
    out_path = Path(f"feature_{feat}_membership_curves.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()

print(f"K (Clusters): {K_detected} | Features: {use_cols}")
if meta:
    print("Meta:", meta)
