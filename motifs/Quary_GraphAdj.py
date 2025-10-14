import networkx as nx
import numpy as np


class DirectedGraph:
    """Directed graph handler for traffic network."""

    def __init__(self, edges_file):
        self.G = self._build_graph(edges_file)

    def _build_graph(self, edges_file):
        """Build directed graph from edges file."""
        with open(edges_file, 'r') as file:
            lines = file.readlines()
            data = [tuple(map(int, line.strip().split(','))) for line in lines]

        graph = nx.DiGraph()
        for start, end in data:
            graph.add_edge(start, end)
        return graph

    def get_adjacency_matrix(self):
        """Get adjacency matrix of the graph."""
        return nx.adjacency_matrix(self.G)

    def get_node_mapping(self):
        """Create node to ID mapping."""
        nodes = list(self.G.nodes())
        return {nodes[i]: i for i in range(len(nodes))}

    def get_k_order_neighborhood(self, node_id, k):
        """Get k-order neighborhood of a node.

        Args:
            node_id: Target node ID
            k: Neighborhood order

        Returns:
            List of neighborhood node IDs
        """
        node_to_id = self.get_node_mapping()
        id_to_node = {v: k for k, v in node_to_id.items()}
        adjacency_matrix = self.get_adjacency_matrix()

        if node_id not in node_to_id:
            raise ValueError(f"Node {node_id} not in graph")
        if k < 0:
            raise ValueError("k must be non-negative integer")

        edge_id = node_to_id[node_id]
        if edge_id < 0 or edge_id >= adjacency_matrix.shape[0]:
            raise ValueError("Invalid edge_id")

        neighborhood_set = {edge_id}
        for _ in range(k):
            current_neighbors = set()
            for node in neighborhood_set:
                neighbors = np.nonzero(adjacency_matrix[node, :])[1]
                current_neighbors.update(neighbors)
            neighborhood_set.update(current_neighbors)

        return [id_to_node[edge_num] for edge_num in neighborhood_set]

    def get_subgraph(self, sub_nodes):
        """Extract subgraph for given nodes."""
        return self.G.subgraph(sub_nodes)