import networkx as nx
import matplotlib.pyplot as plt


#debugging
#print("running")

class CausalGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.weights = {}  # Optional: {(src, dst): weight}

    def add_node(self, node):
        self.graph.add_node(node)

    def add_edge(self, source, target, weight=None):
        self.graph.add_edge(source, target)
        if weight is not None:
            self.weights[(source, target)] = weight

    def update_weight(self, source, target, new_weight):
        if (source, target) in self.graph.edges:
            self.weights[(source, target)] = new_weight

    def estimate_weights(self, data):
        """
        data: Dict[str, List[float]] – z. B. {"angle": [...], "velocity": [...], "reward": [...]}
        Beispiel: lineare Regression zur Schätzung von Einflussgrößen
        """
        from sklearn.linear_model import LinearRegression
        for source, target in self.graph.edges:
            X = [[x] for x in data[source]]
            y = data[target]
            model = LinearRegression().fit(X, y)
            self.weights[(source, target)] = model.coef_[0]

    def intervene(self, node, value):
        """
        Führt eine Intervention do(node=value) aus, indem eingehende Kanten entfernt werden.
        """
        if node in self.graph:
            self.graph.remove_edges_from(list(self.graph.in_edges(node)))
            print(f"Intervention: set {node} := {value} (incoming edges removed)")

    def visualize(self, with_weights=True):
        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', node_size=2000)
        if with_weights:
            labels = {edge: f"{self.weights.get(edge, '?'):.2f}" for edge in self.graph.edges}
            nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=labels)
        plt.show()

    def get_parents(self, node):
        return list(self.graph.predecessors(node))

    def get_children(self, node):
        return list(self.graph.successors(node))
