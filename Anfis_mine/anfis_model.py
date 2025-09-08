import torch
import torch.nn as nn
import torch.nn.functional as F

class ANFIS(nn.Module):
    def __init__(self):
        super(ANFIS, self).__init__()

        self.n_rules = 6
        #risklist = [0.1, 0.4, 0.9]
        

        # Membership params: 2 Inputs x 3 Memberships (neg, null, pos)
        # Für jeden: Zentrum + Breite (sigma)
        self.theta_centers = nn.Parameter(torch.tensor([-0.2, 0.0, 0.2]))
        self.theta_sigmas = nn.Parameter(torch.tensor([0.1, 0.1, 0.1]))

        self.theta_dot_centers = nn.Parameter(torch.tensor([-2.0, 0.0, 2.0]))
        self.theta_dot_sigmas = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # Regelkombinationen (Indexpaare für theta * theta_dot)
        self.rule_indices = [
            (0, 0),  # theta neg, theta_dot neg
            (0, 2),  # theta neg, theta_dot pos
            (1, 0),  # theta null, theta_dot neg
            (1, 2),  # theta null, theta_dot pos
            (2, 0),  # theta pos, theta_dot neg
            (2, 2),  # theta pos, theta_dot pos
        ]

        # Konsequenzparameter: a·theta + b·theta_dot + c
        self.a = nn.Parameter(torch.randn(self.n_rules))
        self.b = nn.Parameter(torch.randn(self.n_rules))
        self.c = nn.Parameter(torch.zeros(self.n_rules))

    def gaussian(self, x, center, sigma):
        return torch.exp(-0.5 * ((x - center) / sigma)**2)

    def forward(self, theta, theta_dot):
        # Batch-fähig: theta, theta_dot ∈ (batch_size,)
        batch_size = theta.size(0)

        # Compute MF outputs (batch_size, 3)
        theta_mf = torch.stack([self.gaussian(theta, c, s)
                                for c, s in zip(self.theta_centers, self.theta_sigmas)], dim=1)

        theta_dot_mf = torch.stack([self.gaussian(theta_dot, c, s)
                                    for c, s in zip(self.theta_dot_centers, self.theta_dot_sigmas)], dim=1)

        # Fuzzy rule firing strengths (batch_size, 6)
        rule_activations = []
        for i, (ti, tdi) in enumerate(self.rule_indices):
            w = theta_mf[:, ti] * theta_dot_mf[:, tdi]
            rule_activations.append(w)
        w = torch.stack(rule_activations, dim=1)  # (batch_size, 6)

        # Regeloutputs
        u = self.a * theta.view(-1, 1) + self.b * theta_dot.view(-1, 1) + self.c  # (batch_size, 6)

        # Output (gewichtetes Mittel)
        weighted_sum = (w * u).sum(dim=1)
        norm_w = w.sum(dim=1) + 1e-6  # vermeide Division durch 0
        output = weighted_sum / norm_w

        return output
