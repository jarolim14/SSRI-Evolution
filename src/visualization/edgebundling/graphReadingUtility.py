import json
from typing import Dict, Tuple, Union
import networkx as nx
import igraph as ig


class graphReadingUtility:
    @staticmethod
    def read_and_clean_graph(path: str) -> ig.Graph:
        g = ig.Graph.Read_GraphML(path)
        g.vs["node_id"] = [int(i) for i in range(g.vcount())]

        if "id" in g.vs.attribute_names():
            g.vs["node_name"] = g.vs["id"]
            del g.vs["id"]

        if "cluster" in g.vs.attribute_names():
            g.vs["cluster"] = [int(cluster) for cluster in g.vs["cluster"]]

        if "year" in g.vs.attribute_names():
            g.vs["year"] = [int(year) for year in g.vs["year"]]

        if "eid" in g.vs.attribute_names():
            del g.vs["eid"]

        if "centrality_alpha0.3_k10_res0.006" in g.vs.attribute_names():
            del g.vs["centrality_alpha0.3_k10_res0.006"]

        if "centrality_alpha0.3_k10_res0.002" in g.vs.attribute_names():
            g.vs["centrality"] = g.vs["centrality_alpha0.3_k10_res0.002"]
            del g.vs["centrality_alpha0.3_k10_res0.002"]

        g.es["edge_id"] = list(range(g.ecount()))
        print("Node Attributes:", g.vs.attribute_names())
        print("Edge Attributes:", g.es.attribute_names())
        # print number of nodes and edges
        print(f"Number of nodes: {g.vcount()}")
        print(f"Number of edges: {g.ecount()}")
        return g

    @staticmethod
    def subgraph_of_clusters(G, clusters):
        if isinstance(G, nx.Graph):
            nodes = [
                node for node in G.nodes if G.nodes[node].get("cluster") in clusters
            ]
            return G.subgraph(nodes)
        elif isinstance(G, ig.Graph):
            nodes = [v.index for v in G.vs if v["cluster"] in clusters]
            return G.subgraph(nodes)
        else:
            raise TypeError("Input must be a NetworkX Graph or an igraph Graph")

    @staticmethod
    def add_cluster_labels(
        G: Union[nx.Graph, ig.Graph],
        labels_file_path: str = "../output/cluster-qualifications/raw_cluster_labels.json",
    ) -> Tuple[Union[nx.Graph, ig.Graph], Dict[float, str]]:
        """
        Add cluster labels to the graph nodes.

        Args:
            G (Union[nx.Graph, ig.Graph]): The input graph (NetworkX or igraph).
            labels_file_path (str): Path to the JSON file containing cluster labels.

        Returns:
            Tuple[Union[nx.Graph, ig.Graph], Dict[float, str]]:
                The graph with added cluster labels and the cluster label dictionary.
        """
        with open(labels_file_path) as file:
            cluster_label_dict = json.load(file)
        cluster_label_dict = {float(k): v[0] for k, v in cluster_label_dict.items()}

        if isinstance(G, nx.Graph):
            for node in G.nodes:
                cluster = G.nodes[node]["cluster"]
                G.nodes[node]["cluster_label"] = cluster_label_dict.get(
                    cluster, "Unknown"
                )
        elif isinstance(G, ig.Graph):
            G.vs["cluster_label"] = [
                cluster_label_dict.get(v["cluster"], "Unknown") for v in G.vs
            ]
        else:
            raise TypeError("Input must be a NetworkX Graph or an igraph Graph")

        return G, cluster_label_dict
