import json
from collections import Counter

import igraph as ig
import numpy as np


class NetworkDescriptives:
    """
    Compute comprehensive statistics for an igraph.Graph object.
    Usage:
        nd = NetworkDescriptives(graph)
        stats = nd.get_stats()
    """

    def __init__(self, graph: ig.Graph):
        """
        Initialize with an igraph.Graph object.

        Parameters
        ----------
        graph : igraph.Graph
            The graph to analyze
        """
        self.graph = graph

    def get_stats(self) -> dict:
        """
        Create comprehensive dictionary with most important graph statistics.
        Handles edge cases and provides robust error handling.

        Returns
        -------
        dict : Dictionary containing comprehensive graph statistics
        """
        graph = self.graph
        # Basic properties
        directed = graph.is_directed()
        n_vertices = graph.vcount()
        n_edges = graph.ecount()

        # Initialize stats dictionary with basic info
        stats = {
            "is_directed": directed,
            "number_of_vertices": n_vertices,
            "number_of_edges": n_edges,
            "is_connected": graph.is_connected(mode="weak" if directed else "strong"),
            "number_of_components": len(
                graph.connected_components(mode="weak" if directed else "strong")
            ),
            "density": graph.density(),
        }

        # Handle empty graphs
        if n_vertices == 0 or n_edges == 0:
            stats.update(
                {
                    "average_degree": 0,
                    "average_in_degree": 0 if directed else None,
                    "average_out_degree": 0 if directed else None,
                    "diameter": None,
                    "average_path_length": None,
                    "transitivity": 0,
                    "modularity": None,
                    "average_clustering": 0,
                    "assortativity": None,
                    "average_edge_weight": None,
                }
            )
            return stats

        # === DEGREE STATISTICS ===
        try:
            if directed:
                in_degrees = graph.indegree()
                out_degrees = graph.outdegree()
                stats["average_in_degree"] = np.mean(in_degrees)
                stats["average_out_degree"] = np.mean(out_degrees)
                stats["max_in_degree"] = max(in_degrees)
                stats["max_out_degree"] = max(out_degrees)
                stats["average_degree"] = np.mean(graph.degree())
            else:
                degrees = graph.degree()
                stats["average_degree"] = np.mean(degrees)
                stats["max_degree"] = max(degrees)
                stats["degree_std"] = np.std(degrees)
                stats["average_in_degree"] = None
                stats["average_out_degree"] = None
        except Exception as e:
            stats["average_degree"] = None
            stats["average_in_degree"] = None
            stats["average_out_degree"] = None

        # === PATH LENGTH STATISTICS ===
        try:
            if stats["is_connected"]:
                stats["diameter"] = graph.diameter(directed=directed)
                stats["average_path_length"] = graph.average_path_length(
                    directed=directed
                )
            else:
                # For disconnected graphs, analyze largest component
                components = graph.connected_components(
                    mode="weak" if directed else "strong"
                )
                largest_comp = max(components, key=len)
                largest_subgraph = graph.subgraph(largest_comp)

                if largest_subgraph.vcount() > 1:
                    stats["diameter"] = largest_subgraph.diameter(directed=directed)
                    stats["average_path_length"] = largest_subgraph.average_path_length(
                        directed=directed
                    )
                    stats["largest_component_size"] = len(largest_comp)
                    stats["largest_component_fraction"] = len(largest_comp) / n_vertices
                else:
                    stats["diameter"] = None
                    stats["average_path_length"] = None
                    stats["largest_component_size"] = len(largest_comp)
                    stats["largest_component_fraction"] = len(largest_comp) / n_vertices
        except Exception as e:
            stats["diameter"] = None
            stats["average_path_length"] = None

        # === CLUSTERING STATISTICS ===
        try:
            if directed:
                stats["transitivity"] = graph.transitivity_directed()
                # For directed graphs, local clustering is more complex
                try:
                    clustering_coeffs = graph.transitivity_local_undirected(mode="zero")
                    valid_clusterings = [
                        c
                        for c in clustering_coeffs
                        if c is not None and not np.isnan(c)
                    ]
                    stats["average_clustering"] = (
                        np.mean(valid_clusterings) if valid_clusterings else 0
                    )
                except:
                    stats["average_clustering"] = None
            else:
                stats["transitivity"] = graph.transitivity_undirected()
                try:
                    clustering_coeffs = graph.transitivity_local_undirected(mode="zero")
                    valid_clusterings = [
                        c
                        for c in clustering_coeffs
                        if c is not None and not np.isnan(c)
                    ]
                    stats["average_clustering"] = (
                        np.mean(valid_clusterings) if valid_clusterings else 0
                    )
                except:
                    stats["average_clustering"] = None
        except Exception as e:
            stats["transitivity"] = None
            stats["average_clustering"] = None

        # === MODULARITY AND COMMUNITY STRUCTURE ===
        try:
            # Check if clusters are already assigned
            if "cluster" in graph.vs.attributes():
                clusters = graph.vs["cluster"]
                partition = ig.VertexClustering(graph, clusters)
                stats["modularity"] = partition.modularity
                stats["number_of_clusters"] = len(set(clusters))

                # Cluster size statistics
                cluster_sizes = Counter(clusters)
                stats["average_cluster_size"] = np.mean(list(cluster_sizes.values()))
                stats["max_cluster_size"] = max(cluster_sizes.values())
                stats["min_cluster_size"] = min(cluster_sizes.values())
                stats["cluster_size_std"] = np.std(list(cluster_sizes.values()))
            else:
                # Compute community detection
                communities = graph.community_multilevel()
                stats["modularity"] = communities.modularity
                stats["number_of_clusters"] = len(communities)

                # Cluster size statistics
                cluster_sizes = [len(community) for community in communities]
                stats["average_cluster_size"] = np.mean(cluster_sizes)
                stats["max_cluster_size"] = max(cluster_sizes)
                stats["min_cluster_size"] = min(cluster_sizes)
                stats["cluster_size_std"] = np.std(cluster_sizes)

        except Exception as e:
            stats["modularity"] = None
            stats["number_of_clusters"] = None
            stats["average_cluster_size"] = None
            stats["max_cluster_size"] = None
            stats["min_cluster_size"] = None
            stats["cluster_size_std"] = None

        # === ASSORTATIVITY ===
        try:
            stats["assortativity"] = graph.assortativity_degree(directed=directed)
        except Exception as e:
            stats["assortativity"] = None

        # === EDGE WEIGHTS ===
        try:
            if "weight" in graph.es.attributes():
                weights = graph.es["weight"]
                stats["average_edge_weight"] = np.mean(weights)
                stats["edge_weight_std"] = np.std(weights)
                stats["max_edge_weight"] = max(weights)
                stats["min_edge_weight"] = min(weights)
            else:
                stats["average_edge_weight"] = None
                stats["edge_weight_std"] = None
                stats["max_edge_weight"] = None
                stats["min_edge_weight"] = None
        except Exception as e:
            stats["average_edge_weight"] = None
            stats["edge_weight_std"] = None
            stats["max_edge_weight"] = None
            stats["min_edge_weight"] = None

        # === CENTRALITY MEASURES ===
        try:
            # Betweenness centrality
            betweenness = graph.betweenness(directed=directed)
            stats["average_betweenness"] = np.mean(betweenness)
            stats["max_betweenness"] = max(betweenness)

            # Centralization index for betweenness
            if n_vertices > 2:
                max_possible = (n_vertices - 1) * (n_vertices - 2) / 2
                if directed:
                    max_possible *= 2
                stats["betweenness_centralization"] = (
                    stats["max_betweenness"] - stats["average_betweenness"]
                ) / max_possible
            else:
                stats["betweenness_centralization"] = 0

        except Exception as e:
            stats["average_betweenness"] = None
            stats["max_betweenness"] = None
            stats["betweenness_centralization"] = None

        try:
            # Closeness centrality (only for connected components)
            if stats["is_connected"]:
                closeness = graph.closeness(mode="all" if directed else "all")
                stats["average_closeness"] = np.mean(closeness)
                stats["max_closeness"] = max(closeness)
            else:
                stats["average_closeness"] = None
                stats["max_closeness"] = None
        except Exception as e:
            stats["average_closeness"] = None
            stats["max_closeness"] = None

        try:
            # Eigenvector centrality
            eigenvector = graph.eigenvector_centrality(directed=directed)
            stats["average_eigenvector"] = np.mean(eigenvector)
            stats["max_eigenvector"] = max(eigenvector)
        except Exception as e:
            stats["average_eigenvector"] = None
            stats["max_eigenvector"] = None

        # === SMALL WORLD PROPERTIES ===
        if (
            stats["average_clustering"] is not None
            and stats["average_path_length"] is not None
            and stats["average_degree"] is not None
        ):
            try:
                # Compare to random graph expectations
                random_clustering = stats["average_degree"] / n_vertices
                random_path_length = (
                    np.log(n_vertices) / np.log(stats["average_degree"])
                    if stats["average_degree"] > 1
                    else None
                )

                if random_path_length and random_clustering > 0:
                    clustering_ratio = stats["average_clustering"] / random_clustering
                    path_ratio = stats["average_path_length"] / random_path_length
                    stats["small_world_sigma"] = clustering_ratio / path_ratio
                    stats["small_world_omega"] = (
                        path_ratio - clustering_ratio
                    )  # Alternative measure
                else:
                    stats["small_world_sigma"] = None
                    stats["small_world_omega"] = None
            except Exception as e:
                stats["small_world_sigma"] = None
                stats["small_world_omega"] = None
        else:
            stats["small_world_sigma"] = None
            stats["small_world_omega"] = None

        # === ADDITIONAL STRUCTURAL PROPERTIES ===
        try:
            # Reciprocity (for directed graphs)
            if directed:
                stats["reciprocity"] = graph.reciprocity()
            else:
                stats["reciprocity"] = None
        except Exception as e:
            stats["reciprocity"] = None

        # Clean up any NaN or infinite values
        for key, value in stats.items():
            if isinstance(value, (float, np.floating)) and (
                np.isnan(value) or np.isinf(value)
            ):
                stats[key] = None

        return stats
