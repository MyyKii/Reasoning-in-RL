import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import json

# ======= Daten laden =======
with open("collected_data.json", "r") as f:
    raw_data = json.load(f)


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


X = np.array([compute_features(d["state"], d["action"]) for d in raw_data], dtype=np.float32)
y = np.array([d["label"] for d in raw_data], dtype=np.float32).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======= Torch-Tensors =======
X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test)

# ======= MLP-Definition =======
class MLP(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.net(x)

model = MLP(input_size=X.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# ======= Training =======
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_train_t)
    loss = criterion(outputs, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            preds = (model(X_test_t) > 0.5).float()
            acc = (preds.eq(y_test_t).sum() / y_test_t.shape[0]).item()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Test-Acc: {acc:.2f}")

# ======= Fertig trainiertes Modell speichern =======
torch.save(model.state_dict(), "mlp_model.pth")
