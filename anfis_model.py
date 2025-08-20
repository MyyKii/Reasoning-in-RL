import torch
from anfis.model import ANFIS

# ANFIS mit 2 Inputs (z.B. Winkel & Winkelgeschwindigkeit), 1 Output (z.B. Kraft)
model = ANFIS(n_inputs=2, n_rules=5)

# Beispielinput
x = torch.tensor([0.1, -0.05])  # z.B. [theta, theta_dot]

# Vorhersage (z. B. Steuerkraft)
y = model(x)

print("Output:", y)
