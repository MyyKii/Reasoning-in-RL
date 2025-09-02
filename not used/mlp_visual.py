from torchviz import make_dot
import torch

# Beispielinput – gleiche Dimension wie beim Training
sample_input = torch.randn(1, INPUT_SIZE)

# Vorwärtspass
output = model(sample_input)

# Netzwerk zeichnen
make_dot(output, params=dict(model.named_parameters())).render("mlp_graph", format="png")
