import numpy as np
import networkx as nx
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pickle
from typing import List, Tuple, Dict, Set


class MotifGraphConstructor:
    """Motif Graph Constructor for Spatio-temporal Traffic Data

    This class implements the algorithm to construct motif-enhanced graphs
    from traffic speed data using Dynamic Time Warping (DTW) similarity
    and topological relationships.
    """

    def __init__(self, speed_matrix_path: str, adj_matrix_path: str,
                 k_neighbors: int = 5, dtw_threshold: float = 0.8):
        """Initialize the Motif Graph Constructor.

        Args:
            speed_matrix_path: Path to the speed matrix file (.npy)
            adj_matrix_path: Path to the adjacency matrix file (.npy)
            k_neighbors: Number of DTW nearest neighbors to consider
            dtw_threshold: DTW similarity threshold for edge creation
        """
        self.speed_matrix = np.load(speed_matrix_path)
        self.adj_matrix_path = adj_matrix_path
        self.k_neighbors = k_neighbors
        self.dtw_threshold = dtw_threshold
        self.original_graph = self._build_original_graph()
        self.motif_graph = nx.DiGraph()

        # Validate input dimensions
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate input matrices and parameters."""
        if self.speed_matrix.ndim != 2:
            raise ValueError("Speed matrix must be 2-dimensional")

        adj_matrix = np.load(self.adj_matrix_path)
        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Adjacency matrix must be square")

        if self.speed_matrix.shape[1] != adj_matrix.shape[0]:
            raise ValueError("Speed matrix and adjacency matrix must have same number of nodes")

        if not (0 < self.dtw_threshold <= 1):
            raise ValueError("DTW threshold must be in range (0, 1]")

        if self.k_neighbors <= 0:
            raise ValueError("Number of neighbors must be positive")

    def _build_original_graph(self) -> nx.DiGraph:
        """Build the original directed graph from adjacency matrix.

        Returns:
            networkx.DiGraph: Original graph structure
        """
        adj_matrix = np.load(self.adj_matrix_path)
        graph = nx.DiGraph()
        n_nodes = adj_matrix.shape[0]

        for i in range(n_nodes):
            for j in range(n_nodes):
                if adj_matrix[i, j] != 0:
                    graph.add_edge(i, j)

        return graph

    def _calculate_dtw_similarity(self, ts1: np.ndarray, ts2: np.ndarray) -> float:
        """Calculate DTW-based similarity between two time series.

        Args:
            ts1: First time series
            ts2: Second time series

        Returns:
            float: Similarity score between 0 and 1
        """
        distance, _ = fastdtw(ts1, ts2, dist=euclidean)
        return 1.0 / (1.0 + distance)

    def _get_dtw_neighbors(self, node_id: int, candidate_nodes: List[int],
                           order: int = 1) -> List[int]:
        """Get DTW-based neighbors for a given node.

        Args:
            node_id: Target node ID
            candidate_nodes: List of candidate nodes for neighborhood
            order: Neighborhood order (for logging purposes)

        Returns:
            List[int]: List of DTW neighbor nodes
        """
        if node_id >= self.speed_matrix.shape[0]:
            return []

        node_timeseries = self.speed_matrix[node_id]
        similarities = []

        for candidate in candidate_nodes:
            if candidate >= self.speed_matrix.shape[0]:
                continue

            candidate_timeseries = self.speed_matrix[candidate]
            similarity = self._calculate_dtw_similarity(node_timeseries, candidate_timeseries)

            if similarity >= self.dtw_threshold:
                similarities.append((candidate, similarity))

        # Sort by similarity and return top k neighbors
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [node for node, _ in similarities[:self.k_neighbors]]

    def _get_multi_order_neighbors(self, node_id: int) -> Tuple[List[int], List[int], List[int]]:
        """Get multi-order topological neighbors using DTW similarity.

        Args:
            node_id: Target node ID

        Returns:
            Tuple containing:
                - 1st-order neighbors
                - 2nd-order neighbors
                - 3rd-order neighbors
        """
        # Get 1st-order topological neighbors
        first_order_candidates = list(self.original_graph.neighbors(node_id))
        first_order_neighbors = self._get_dtw_neighbors(node_id, first_order_candidates, 1)

        # Get 2nd-order topological neighbors
        second_order_candidates = set()
        for neighbor in first_order_candidates:
            second_order_candidates.update(self.original_graph.neighbors(neighbor))
        second_order_candidates = list(second_order_candidates - {node_id})
        second_order_neighbors = self._get_dtw_neighbors(node_id, second_order_candidates, 2)

        # Get 3rd-order topological neighbors
        third_order_candidates = set()
        for neighbor in second_order_candidates:
            third_order_candidates.update(self.original_graph.neighbors(neighbor))
        third_order_candidates = list(third_order_candidates - {node_id})
        third_order_neighbors = self._get_dtw_neighbors(node_id, third_order_candidates, 3)

        return first_order_neighbors, second_order_neighbors, third_order_neighbors

    def _detect_line_motif(self, nodes: List[int]) -> bool:
        """Detect if nodes form a linear motif.

        Args:
            nodes: List of node IDs

        Returns:
            bool: True if nodes form a linear motif
        """
        if len(nodes) < 2:
            return False

        for i in range(len(nodes) - 1):
            if not self.original_graph.has_edge(nodes[i], nodes[i + 1]):
                return False
        return True

    def _detect_triangle_motif(self, nodes: List[int]) -> bool:
        """Detect if three nodes form a triangle motif.

        Args:
            nodes: List of three node IDs

        Returns:
            bool: True if nodes form a triangle motif
        """
        if len(nodes) != 3:
            return False

        a, b, c = nodes
        return (self.original_graph.has_edge(a, b) and
                self.original_graph.has_edge(b, c) and
                self.original_graph.has_edge(c, a))

    def _detect_quadrilateral_motif(self, nodes: List[int]) -> bool:
        """Detect if four nodes form a quadrilateral motif.

        Args:
            nodes: List of four node IDs

        Returns:
            bool: True if nodes form a quadrilateral motif
        """
        if len(nodes) != 4:
            return False

        a, b, c, d = nodes
        return (self.original_graph.has_edge(a, b) and
                self.original_graph.has_edge(b, c) and
                self.original_graph.has_edge(c, d) and
                self.original_graph.has_edge(d, a))

    def _detect_diagonal_motif(self, nodes: List[int]) -> bool:
        """Detect if quadrilateral has diagonal connections.

        Args:
            nodes: List of four node IDs

        Returns:
            bool: True if diagonal connections exist
        """
        if len(nodes) != 4:
            return False

        a, b, c, d = nodes
        return (self.original_graph.has_edge(b, d) or
                self.original_graph.has_edge(d, b))

    def _add_motif_to_graph(self, motif_type: str, nodes: List[int]) -> None:
        """Add detected motif to the motif graph.

        Args:
            motif_type: Type of motif ('line', 'triangle', 'quadrilateral', 'diagonal')
            nodes: List of nodes in the motif
        """
        if len(nodes) < 2:
            return

        motif_id = f"{motif_type}_{'_'.join(map(str, sorted(nodes)))}"

        # Add nodes to graph
        for node in nodes:
            self.motif_graph.add_node(node, node_type='traffic')

        # Add edges based on motif type
        if motif_type == 'line':
            for i in range(len(nodes) - 1):
                self.motif_graph.add_edge(nodes[i], nodes[i + 1],
                                          motif_type='line',
                                          motif_id=motif_id)
        elif motif_type == 'triangle':
            for i in range(len(nodes)):
                j = (i + 1) % len(nodes)
                self.motif_graph.add_edge(nodes[i], nodes[j],
                                          motif_type='triangle',
                                          motif_id=motif_id)
        elif motif_type == 'quadrilateral':
            for i in range(len(nodes)):
                j = (i + 1) % len(nodes)
                self.motif_graph.add_edge(nodes[i], nodes[j],
                                          motif_type='quadrilateral',
                                          motif_id=motif_id)
        elif motif_type == 'diagonal':
            if len(nodes) == 4:
                self.motif_graph.add_edge(nodes[1], nodes[3],
                                          motif_type='diagonal',
                                          motif_id=motif_id)

    def construct_motif_graph(self) -> nx.DiGraph:
        """Construct motif-enhanced graph from traffic data.

        Returns:
            networkx.DiGraph: Motif-enhanced graph structure
        """
        print("Starting motif graph construction...")

        total_nodes = len(self.original_graph.nodes())
        processed_nodes = 0

        for node_id in self.original_graph.nodes():
            try:
                # Get multi-order DTW neighbors
                first_order_neighbors, second_order_neighbors, third_order_neighbors = \
                    self._get_multi_order_neighbors(node_id)

                # Process 1st-order neighbors
                for neighbor_1 in first_order_neighbors:
                    if self._detect_line_motif([node_id, neighbor_1]):
                        self._add_motif_to_graph('line', [node_id, neighbor_1])

                # Process 2nd-order neighbors
                for neighbor_1 in first_order_neighbors:
                    for neighbor_2 in second_order_neighbors:
                        # Check linear motif
                        if self._detect_line_motif([node_id, neighbor_1, neighbor_2]):
                            self._add_motif_to_graph('line', [node_id, neighbor_1, neighbor_2])

                        # Check triangle motif
                        if self._detect_triangle_motif([node_id, neighbor_1, neighbor_2]):
                            self._add_motif_to_graph('triangle', [node_id, neighbor_1, neighbor_2])

                # Process 3rd-order neighbors
                for neighbor_1 in first_order_neighbors:
                    for neighbor_2 in second_order_neighbors:
                        for neighbor_3 in third_order_neighbors:
                            # Check linear motif
                            if self._detect_line_motif([node_id, neighbor_1, neighbor_2, neighbor_3]):
                                self._add_motif_to_graph('line', [node_id, neighbor_1, neighbor_2, neighbor_3])

                            # Check quadrilateral motif
                            if self._detect_quadrilateral_motif([node_id, neighbor_1, neighbor_2, neighbor_3]):
                                self._add_motif_to_graph('quadrilateral', [node_id, neighbor_1, neighbor_2, neighbor_3])

                            # Check diagonal motif
                            if self._detect_diagonal_motif([node_id, neighbor_1, neighbor_2, neighbor_3]):
                                self._add_motif_to_graph('diagonal', [node_id, neighbor_1, neighbor_2, neighbor_3])

                processed_nodes += 1
                if processed_nodes % 100 == 0:
                    print(f"Processed {processed_nodes}/{total_nodes} nodes...")

            except Exception as e:
                print(f"Error processing node {node_id}: {str(e)}")
                continue

        print(f"Motif graph construction completed. "
              f"Graph contains {self.motif_graph.number_of_nodes()} nodes and "
              f"{self.motif_graph.number_of_edges()} edges")

        return self.motif_graph

    def get_adjacency_matrix(self) -> np.ndarray:
        """Generate adjacency matrix from motif graph.

        Returns:
            numpy.ndarray: Adjacency matrix of the motif graph
        """
        nodes = sorted(self.motif_graph.nodes())
        node_index = {node: idx for idx, node in enumerate(nodes)}

        n_nodes = len(nodes)
        adj_matrix = np.zeros((n_nodes, n_nodes))

        for edge in self.motif_graph.edges():
            i, j = edge
            adj_matrix[node_index[i], node_index[j]] = 1

        return adj_matrix

    def save_graph(self, graph_path: str, adj_matrix_path: str) -> None:
        """Save motif graph and adjacency matrix to files.

        Args:
            graph_path: Path to save the graph object
            adj_matrix_path: Path to save the adjacency matrix
        """
        # Save graph object
        with open(graph_path, 'wb') as f:
            pickle.dump(self.motif_graph, f)

        # Save adjacency matrix
        adj_matrix = self.get_adjacency_matrix()
        np.save(adj_matrix_path, adj_matrix)

        print(f"Motif graph saved to {graph_path}")
        print(f"Adjacency matrix saved to {adj_matrix_path}")

    def analyze_motifs(self) -> Dict[str, int]:
        """Analyze distribution of different motif types in the graph.

        Returns:
            Dict[str, int]: Count of each motif type
        """
        motif_counts = {
            'line': 0,
            'triangle': 0,
            'quadrilateral': 0,
            'diagonal': 0
        }

        for _, _, edge_data in self.motif_graph.edges(data=True):
            motif_type = edge_data.get('motif_type', 'unknown')
            if motif_type in motif_counts:
                motif_counts[motif_type] += 1

        print("Motif distribution analysis:")
        for motif_type, count in motif_counts.items():
            print(f"  {motif_type}: {count} edges")

        return motif_counts


def main():
    """Main execution function for motif graph construction."""
    # Configuration parameters
    config = {
        'speed_matrix_path': '../Dataset/Pr_Hi.npy',
        'adj_matrix_path': '../Dataset/adj_matrix.npy',
        'output_graph_path': '../Dataset/motif_graph.pkl',
        'output_adj_path': '../Dataset/motif_adj_matrix.npy',
        'k_neighbors': 5,
        'dtw_threshold': 0.7
    }

    try:
        # Initialize constructor
        constructor = MotifGraphConstructor(
            speed_matrix_path=config['speed_matrix_path'],
            adj_matrix_path=config['adj_matrix_path'],
            k_neighbors=config['k_neighbors'],
            dtw_threshold=config['dtw_threshold']
        )

        # Construct motif graph
        motif_graph = constructor.construct_motif_graph()

        # Analyze motif distribution
        motif_distribution = constructor.analyze_motifs()

        # Save results
        constructor.save_graph(
            graph_path=config['output_graph_path'],
            adj_matrix_path=config['output_adj_path']
        )

        print("Motif graph construction completed successfully!")

    except Exception as e:
        print(f"Error in motif graph construction: {str(e)}")
        raise


if __name__ == "__main__":
    main()