import time
import networkx as nx
import igraph as ig
from typing import Dict, Optional, Tuple, Union


class frLayoutUtility:
    """
    Layout utility class for igraph layout operations. made for fruchterman-reingold layout.

    Args:
        g (Union[nx.Graph, ig.Graph]): The input graph (NetworkX or igraph).
        layout_params (Optional[Dict]): The layout parameters.

    Returns:
        Tuple[nx.Graph, Dict]: The graph with assigned coordinates and the layout dictionary.
    """

    @staticmethod
    def fr_layout_nx(
        g: Union[nx.Graph, ig.Graph], layout_params: Optional[Dict] = None
    ) -> Tuple[nx.Graph, Dict]:
        print("Starting Fruchterman-Reingold layout process...")
        start_time = time.time()

        if layout_params is None:
            layout_params = {
                "iterations": 100,
                "threshold": 0.00001,
                "weight": "weight",
                "scale": 1,
                "center": (0, 0),
                "dim": 2,
                "seed": 1887,
            }
        print(f"Layout parameters: {layout_params}")

        if not isinstance(g, nx.Graph):
            print("Converting to NetworkX Graph...")
            G = g.to_networkx()
            print("Conversion complete.")
        else:
            G = g

        print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

        print("Calculating layout...")
        layout_start_time = time.time()
        pos = nx.spring_layout(G, **layout_params)
        layout_end_time = time.time()
        print(
            f"Layout calculation completed in {layout_end_time - layout_start_time:.2f} seconds."
        )

        print("Processing layout results...")
        node_xy_dict = {node: pos[node] for node in G.nodes}

        x_values, y_values = zip(*node_xy_dict.values())
        min_x, max_x = min(x_values), max(x_values)
        min_y, max_y = min(y_values), max(y_values)

        print(f"Layout boundaries:")
        print(f"X-axis: Min = {min_x:.2f}, Max = {max_x:.2f}")
        print(f"Y-axis: Min = {min_y:.2f}, Max = {max_y:.2f}")

        print("Assigning coordinates to nodes...")
        for node in G.nodes:
            G.nodes[node]["x"] = node_xy_dict[node][0]
            G.nodes[node]["y"] = node_xy_dict[node][1]

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Layout process completed in {total_time:.2f} seconds.")

        return G, pos
