import igraph as ig
import numpy as np
import random
import networkx as nx  # Import NetworkX for conversion
from typing import Dict


class pruneEdges:
    """
    A class for pruning edges in a graph based on specific criteria, such as edge weight percentiles or random selection.
    """

    def __init__(self, g):
        self.g = self._to_igraph(g)
        self.g_pruned = None
        self.initial_edge_count = 0
        self.initial_isolates = 0
        self.final_edge_count = 0
        self.final_isolates = 0

    def _to_igraph(self, g):
        """
        Convert a NetworkX graph to an igraph Graph if necessary.
        """
        if isinstance(g, nx.Graph):
            g = ig.Graph.from_networkx(g)
        return g

    def _update_statistics(self) -> None:
        """
        Update class attributes related to graph statistics.
        """
        self.initial_edge_count = self.g.ecount()
        self.initial_isolates = len(self.g.vs.select(_degree=0))
        self.final_edge_count = self.g_pruned.ecount()
        self.final_isolates = len(self.g_pruned.vs.select(_degree=0))

    def prune_edges_by_weight_percentile(self, percentile: float) -> ig.Graph:
        """
        Keep edges from the graph that have weight greater than or equal to the specified percentile weight.

        Args:
            percentile (float): The percentile to use as the threshold for keeping edges (between 0 and 1).

        Returns:
            ig.Graph: A new graph with edges kept based on the specified percentile.

        Raises:
            ValueError: If the input graph has no 'weight' attribute for edges.
            ValueError: If percentile is not between 0 and 1.
        """
        if not (0 <= percentile <= 1):
            raise ValueError("Percentile must be between 0 and 1.")

        if "weight" not in self.g.es.attributes():
            raise ValueError("Input graph must have a 'weight' attribute for edges.")

        # Get the weight threshold for the given percentile
        weights = self.g.es["weight"]
        weight_threshold = np.percentile(weights, (1 - percentile) * 100)

        # Find edges to keep based on weight
        edges_to_keep = [
            edge.index for edge in self.g.es if edge["weight"] >= weight_threshold
        ]
        threshold_edges = [
            edge.index for edge in self.g.es if edge["weight"] == weight_threshold
        ]

        print(f"Weight Threshold: {weight_threshold:.2f}")
        print(f"Edges with this weight: {len(threshold_edges)}")

        # Ensure the correct number of edges are kept
        target_edge_count = int(self.g.ecount() * percentile)
        edges_to_add = target_edge_count - len(edges_to_keep)

        if edges_to_add > 0:
            random.shuffle(threshold_edges)
            edges_to_keep.extend(threshold_edges[:edges_to_add])

        self.g_pruned = self.g.subgraph_edges(edges_to_keep, delete_vertices=False)

        # Update statistics
        self._update_statistics()
        print(f"Kept top {percentile:.1%} of edges by weight")
        print(f"Edges kept: {self.final_edge_count} out of {self.initial_edge_count}")

        return self.g_pruned

    def prune_edges_randomly(self, percentile: float) -> ig.Graph:
        """
        Keep a random selection of edges based on the provided percentile.

        Args:
            percentile (float): The percentile of edges to keep (between 0 and 1).

        Returns:
            ig.Graph: A new graph with the randomly selected edges.

        Raises:
            ValueError: If percentile is not between 0 and 1.
        """
        if not (0 <= percentile <= 1):
            raise ValueError("Percentile must be between 0 and 1.")

        total_edges = self.g.ecount()
        edges_to_keep_count = int(total_edges * percentile)

        # Randomly select edges to keep
        all_edges = list(range(total_edges))
        random.shuffle(all_edges)
        edges_to_keep = all_edges[:edges_to_keep_count]

        self.g_pruned = self.g.subgraph_edges(edges_to_keep, delete_vertices=False)

        # Update statistics
        self._update_statistics()
        print(f"Kept {percentile:.1%} of edges randomly")
        print(f"Edges kept: {self.final_edge_count} out of {self.initial_edge_count}")

        return self.g_pruned

    def get_prune_summary(self) -> Dict[str, int]:
        """
        Returns a summary of the pruning process.

        Returns:
            Dict[str, int]: A dictionary containing the initial and final edge counts and the number of isolates.
        """
        return {
            "initial_edge_count": self.initial_edge_count,
            "final_edge_count": self.final_edge_count,
            "initial_isolates": self.initial_isolates,
            "final_isolates": self.final_isolates,
        }
