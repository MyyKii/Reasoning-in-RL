import numpy as np
import matplotlib.pyplot as plt

def mu_theta_neg(theta):
    if theta >= -0.05:
        return 0.0
    elif theta <= -0.2:
        return 1.0
    else:
        return (-theta - 0.05) / 0.15

def mu_theta_null(theta):
    if abs(theta) >= 0.1:
        return 0.0
    else:
        return 1.0 - abs(theta) / 0.1

def mu_theta_pos(theta):
    if theta <= 0.05:
        return 0.0
    elif theta >= 0.2:
        return 1.0
    else:
        return (theta - 0.05) / 0.15

def mu_theta_dot_neg(theta_dot):
    if theta_dot >= 0.0:
        return 0.0
    elif theta_dot <= -2.0:
        return 1.0
    else:
        return (-theta_dot) / 2.0

def mu_theta_dot_pos(theta_dot):
    if theta_dot <= 0.0:
        return 0.0
    elif theta_dot >= 2.0:
        return 1.0
    else:
        return theta_dot / 2.0

theta_vals = np.linspace(-0.3, 0.3, 500)
theta_dot_vals = np.linspace(-3, 3, 500)

mu_theta_neg_vals = [mu_theta_neg(t) for t in theta_vals]
mu_theta_null_vals = [mu_theta_null(t) for t in theta_vals]
mu_theta_pos_vals = [mu_theta_pos(t) for t in theta_vals]

mu_theta_dot_neg_vals = [mu_theta_dot_neg(td) for td in theta_dot_vals]
mu_theta_dot_pos_vals = [mu_theta_dot_pos(td) for td in theta_dot_vals]

plt.figure()
plt.plot(theta_vals, mu_theta_neg_vals, label="mu_theta_neg")
plt.plot(theta_vals, mu_theta_null_vals, label="mu_theta_null")
plt.plot(theta_vals, mu_theta_pos_vals, label="mu_theta_pos")
plt.title("Fuzzy Membership Functions for θ (Angle)")
plt.xlabel("θ (rad)")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
plt.plot(theta_dot_vals, mu_theta_dot_neg_vals, label="mu_theta_dot_neg")
plt.plot(theta_dot_vals, mu_theta_dot_pos_vals, label="mu_theta_dot_pos")
plt.title("Fuzzy Membership Functions for θ̇ (Angular Velocity)")
plt.xlabel("θ̇ (rad/s)")
plt.ylabel("Membership Degree")
plt.legend()
plt.grid(True)
plt.show()
