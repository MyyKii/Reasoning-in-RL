# main.py oder Testskript
from causal_graph import CausalGraph

cg = CausalGraph()
cg.add_node("angle")
cg.add_node("velocity")
cg.add_node("reward")
cg.add_edge("angle", "velocity")
cg.add_edge("velocity", "reward")

# Optional: Gewichte lernen
data = {
    "angle": [0.1, 0.2, 0.3],
    "velocity": [0.2, 0.4, 0.5],
    "reward": [1, 0.8, 0.3],
}
cg.estimate_weights(data)

# Visualisieren
cg.visualize()

# Intervention durchführen
cg.intervene("velocity", value=0.0)
