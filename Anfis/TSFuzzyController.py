import numpy as np

class TSFuzzyController:
    def __init__(self):
        self.rules = [
            (-10, -2, 0),  # R1: theta neg, theta_dot neg
            (-8,  -1, 0),  # R2: theta neg, theta_dot pos
            (-2,  -3, 0),  # R3: theta null, theta_dot neg
            (-2,  -3, 0),  # R4: theta null, theta_dot pos
            (-8,  -1, 0),  # R5: theta pos, theta_dot neg
            (-10, -2, 0),  # R6: theta pos, theta_dot pos
        ]

    #  Fuzzy Membership for theta = angle of pole 
    def mu_theta_neg(self, theta):
        if theta >= -0.05:
            return 0.0
        elif theta <= -0.2:
            return 1.0
        else:
            return (-theta - 0.05) / (0.15)

    def mu_theta_null(self, theta):
        if abs(theta) >= 0.1:
            return 0.0
        else:
            return 1.0 - abs(theta) / 0.1

    def mu_theta_pos(self, theta):
        if theta <= 0.05:
            return 0.0
        elif theta >= 0.2:
            return 1.0
        else:
            return (theta - 0.05) / 0.15



    #  Fuzzy Membership for theta_dot = angular velocity of pole
    def mu_theta_dot_neg(self, theta_dot):
        if theta_dot >= 0.0:
            return 0.0
        elif theta_dot <= -2.0:
            return 1.0
        else:
            return (-theta_dot) / 2.0

    def mu_theta_dot_pos(self, theta_dot):
        if theta_dot <= 0.0:
            return 0.0
        elif theta_dot >= 2.0:
            return 1.0
        else:
            return theta_dot / 2.0

    def compute(self, theta, theta_dot):
        weights = [
            self.mu_theta_neg(theta) * self.mu_theta_dot_neg(theta_dot),
            self.mu_theta_neg(theta) * self.mu_theta_dot_pos(theta_dot),
            self.mu_theta_null(theta) * self.mu_theta_dot_neg(theta_dot),
            self.mu_theta_null(theta) * self.mu_theta_dot_pos(theta_dot),
            self.mu_theta_pos(theta) * self.mu_theta_dot_neg(theta_dot),
            self.mu_theta_pos(theta) * self.mu_theta_dot_pos(theta_dot),
        ]

        outputs = []
        for i, (a, b, c) in enumerate(self.rules):
            u = a * theta + b * theta_dot + c
            outputs.append(u)

        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0  
        weighted_sum = sum(w * u for w, u in zip(weights, outputs))
        return weighted_sum / total_weight
