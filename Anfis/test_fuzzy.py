from Anfis.TSFuzzyController import TSFuzzyController as Controller


controller = Controller()

theta = 0.12        
theta_dot = -0.7 

u = controller.compute(theta, theta_dot)

print(f"Theta: {theta:.3f}, Theta_dot: {theta_dot:.3f} → Force u = {u:.3f}")
