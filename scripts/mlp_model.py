import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import json
import logging
from wandb_utils import init_wandb_run, log_metrics, finish_wandb_run


MLP_MODEL_PATH = "models/mlp_model.pth"
TRAINING_DATA_PATH = "data/mlp_training_data.json"
EPOCHS = 1000
LEARNING_RATE = 1e-3

#TODO: Important notice: mlp_data_collector creates binary labels now. 
#TODO: Can be changed but needs adjustments in mlp_model if desired. 


# Gymnasium InvertedPendulum action range is [-3, 3]
ACTION_MAX = 3.0
THETA_LIMIT = 0.2  # env upright/termination scale

def compute_features(state, action):
    """
    state order: [x, theta, x_dot, theta_dot]
    action: float
    returns: np.array([f1..f5], dtype=float32)
    """
    x, theta, x_dot, theta_dot = [float(v) for v in state[:4]]
    a = float(action)

    f1 = abs(x) / 1.0
    f2 = abs(theta) / THETA_LIMIT
    f3 = abs(x_dot) / 5.0
    f4 = abs(theta_dot) / 5.0
    f5 = abs(a) / ACTION_MAX

    feats = np.array([f1, f2, f3, f4, f5], dtype=np.float32)
    return np.clip(feats, 0.0, 10.0)



def run_mlp_model():
    logger = logging.getLogger(__name__)

    # ======= load data =======
    with open(TRAINING_DATA_PATH, "r") as f:
        raw_data = json.load(f)

    logger.info(f"Load Data from {TRAINING_DATA_PATH}")
    logger.info(f"Number of Samples in file: {len(raw_data)}")

    logger.info("Starting to train MLP model for risk prediction...")

    X = np.array(
        [compute_features(d["state"], d["action"]) for d in raw_data], dtype=np.float32
    )
    y = np.array([d["label"] for d in raw_data], dtype=np.float32).reshape(-1, 1)

    logger.info(f"Feature-Matrix X Shape: {X.shape}")
    logger.info(f"Label-Vector y Shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42
    )

    # ======= Torch-Tensors =======
    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t = torch.tensor(X_test)
    y_test_t = torch.tensor(y_test)

    logger.info(f"Train-Samples: {len(X_train)}")
    logger.info(f"Test-Samples : {len(X_test)}")

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
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.net(x)

    model = MLP(input_size=X.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    logger.info(f"Number of Epochs: {EPOCHS}")

    config = {
        "training_data_path": TRAINING_DATA_PATH,
        "num_samples": int(X.shape[0]),
        "input_dim": int(X.shape[1]),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": len(X_train),
        "batch_type": "Full-Batch",
        "model_type": "MLP",
        "hidden_layers": [16, 8],
        "hidden_activation": "ReLU",
        "output_dim": 1,
        "output_activation": "Sigmoid",
        "loss_type": "BCELoss",
        "optimizer": "Adam",
    }

    init_wandb_run(
        project="counterfactual-agents",
        job_type="mlp_training",
        config=config,
        run_name="mlp_training_v1",
    )
    logger.info("W&B Run für MLP-Training initialisiert.")

    # ======= Training =======
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                preds = (model(X_test_t) > 0.5).float()
                acc = (preds.eq(y_test_t).sum() / y_test_t.shape[0]).item()
            log_metrics(
                {
                    "test_accuracy": float(acc),
                    "train_loss": float(loss.item()),
                },
                step=epoch + 1,
            )
            logger.info(
                f"Epoche {epoch + 1} – Train-Loss: {loss.item():.4f} – Test-Acc: {acc:.4f}"
            )

    log_metrics({"final_accuracy": float(acc)})
    torch.save(model.state_dict(), MLP_MODEL_PATH)
    logger.info(f"MLP model saved to {MLP_MODEL_PATH}")
    finish_wandb_run()
    logger.info("W&B Run beendet.")


# --- Inference utilities (für Live-Run / Counterfactual-Agent) ------------------

class RiskMLP(nn.Module):
    """Architektur muss exakt der Trainings-Definition entsprechen."""
    def __init__(self, input_size: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


def load_risk_mlp(model_path: str = MLP_MODEL_PATH, device: str = "cpu") -> RiskMLP:
    model = RiskMLP(input_size=5)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_risk(model: RiskMLP, state, action: float) -> float:
    """
    Returns risk in [0,1].
    state: array-like (wie in mlp_training_data.json gespeichert)
    action: float
    """
    feats = compute_features(state, action).astype(np.float32)
    x = torch.from_numpy(feats).unsqueeze(0)  # (1,5)
    with torch.no_grad():
        y = model(x).reshape(-1)[0].item()
    return float(y)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_mlp_model()
