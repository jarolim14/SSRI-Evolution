from datashader.bundling import hammer_bundle
import igraph as ig
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree
from typing import Dict, List, Tuple, Union
from tqdm import tqdm
import time


class OptimizedGraphProcessor:
    def __init__(self, input_path: str, centrality_multiplier: float = -100):
        self.g = self.read_and_clean_graph(input_path)
        self.centrality_multiplier = centrality_multiplier

    @staticmethod
    def read_and_clean_graph(path: str) -> ig.Graph:
        g = ig.Graph.Read_GraphML(path)
        g.vs["node_id"] = range(g.vcount())
        if "id" in g.vs.attribute_names():
            g.vs["node_name"] = g.vs["id"]
            del g.vs["id"]
        g.vs["cluster"] = [int(cluster) for cluster in g.vs["cluster"]]
        g.vs["year"] = [int(year) for year in g.vs["year"]]
        if "centrality_alpha0.3_k10_res0.002" in g.vs.attribute_names():
            g.vs["centrality"] = g.vs["centrality_alpha0.3_k10_res0.002"]
            del g.vs["centrality_alpha0.3_k10_res0.002"]
        g.es["edge_id"] = range(g.ecount())
        return g

    def apply_layout(self):
        layout = self.g.layout_fruchterman_reingold(
            weights="weight", niter=100, seed=1887
        )
        self.g.vs["x"], self.g.vs["y"] = zip(*layout)

    def prune_edges(self):
        weights = self.g.es["weight"]
        median_weight = np.median(weights)
        self.g.es.select(weight_le=median_weight).delete()

    def add_z_coordinate(self):
        for v in self.g.vs:
            v["z"] = v["centrality"] * self.centrality_multiplier

    def create_edge_df(self) -> pd.DataFrame:
        edges = [(e.source, e.target, e["edge_id"], e["weight"]) for e in self.g.es]
        return pd.DataFrame(edges, columns=["source", "target", "edge_id", "weight"])

    def interpolate_edge_points(self, edge_df: pd.DataFrame) -> pd.DataFrame:
        def interpolate(row):
            source = self.g.vs[row["source"]]
            target = self.g.vs[row["target"]]
            t = np.linspace(0, 1, 10)  # Adjust number of points as needed
            x = np.interp(t, [0, 1], [source["x"], target["x"]])
            y = np.interp(t, [0, 1], [source["y"], target["y"]])
            z = np.interp(t, [0, 1], [source["z"], target["z"]])
            return pd.Series({"x": x, "y": y, "z": z})

        return edge_df.apply(interpolate, axis=1)

    def apply_hammer_bundle(self, edge_df: pd.DataFrame) -> pd.DataFrame:
        df_nodes = pd.DataFrame(
            {"x": self.g.vs["x"], "y": self.g.vs["y"], "cluster": self.g.vs["cluster"]}
        )
        bundled_edges = hammer_bundle(
            df_nodes, edge_df, decay=0.90, initial_bandwidth=0.10, iterations=15
        )
        return pd.DataFrame(bundled_edges, columns=["x", "y", "edge_id", "weight"])

    def apply_3d_bundling(
        self,
        edges: pd.DataFrame,
        iterations: int = 10,
        step_size: float = 0.1,
        neighbor_radius: float = 0.1,
        compatibility_threshold: float = 0.6,
        smoothing_iterations: int = 2,
    ) -> pd.DataFrame:

        all_points = np.vstack(
            [
                np.column_stack((edge["x"], edge["y"], edge["z"]))
                for _, edge in edges.iterrows()
            ]
        )
        point_to_edge = np.repeat(np.arange(len(edges)), edges["x"].apply(len))
        tree = cKDTree(all_points)

        for _ in tqdm(range(iterations), desc="Applying 3D bundling"):
            new_points = all_points.copy()

            # Force calculation
            neighbors = tree.query_ball_point(all_points, r=neighbor_radius)
            for i, (point, point_neighbors) in enumerate(zip(all_points, neighbors)):
                if (
                    1 < i < len(all_points) - 1
                ):  # Skip first and last points of each edge
                    edge_index = point_to_edge[i]
                    compatible_neighbors = [
                        n for n in point_neighbors if point_to_edge[n] != edge_index
                    ]
                    if compatible_neighbors:
                        force = np.mean(
                            all_points[compatible_neighbors] - point, axis=0
                        )
                        new_points[i] += step_size * force

            all_points = new_points

            # Smoothing
            edge_lengths = edges["x"].apply(len).values
            edge_starts = np.cumsum(np.insert(edge_lengths, 0, 0))[:-1]
            for _ in range(smoothing_iterations):
                for start, length in zip(edge_starts, edge_lengths):
                    if length > 2:
                        all_points[start + 1 : start + length - 1] = 0.5 * all_points[
                            start + 1 : start + length - 1
                        ] + 0.25 * (
                            all_points[start : start + length - 2]
                            + all_points[start + 2 : start + length]
                        )

        # Update edges DataFrame with bundled coordinates
        start = 0
        for i, length in enumerate(edge_lengths):
            edges.loc[i, "x"] = all_points[start : start + length, 0]
            edges.loc[i, "y"] = all_points[start : start + length, 1]
            edges.loc[i, "z"] = all_points[start : start + length, 2]
            start += length

        return edges

    def process_graph(
        self, use_hammer_bundle: bool = False
    ) -> Tuple[ig.Graph, pd.DataFrame]:
        self.apply_layout()
        self.prune_edges()
        self.add_z_coordinate()
        edge_df = self.create_edge_df()

        if use_hammer_bundle:
            hammer_bundled_edges = self.apply_hammer_bundle(edge_df)
            interpolated_edges = self.interpolate_edge_points(hammer_bundled_edges)
        else:
            interpolated_edges = self.interpolate_edge_points(edge_df)

        bundled_edges = self.apply_3d_bundling(interpolated_edges)
        return self.g, bundled_edges

    @staticmethod
    def save_to_json(
        g: ig.Graph, bundled_edges: pd.DataFrame, nodes_path: str, edges_path: str
    ):
        nodes_json = [
            {
                "id": v["node_id"],
                "name": v["node_name"],
                "x": v["x"],
                "y": v["y"],
                "z": v["z"],
                "cluster": v["cluster"],
                "centrality": v["centrality"],
            }
            for v in g.vs
        ]

        edges_json = [
            {
                "id": int(edge["edge_id"]),
                "source": int(g.es[int(edge["edge_id"])].source),
                "target": int(g.es[int(edge["edge_id"])].target),
                "weight": float(edge["weight"]),
                "points": [
                    {"x": float(x), "y": float(y), "z": float(z)}
                    for x, y, z in zip(edge["x"], edge["y"], edge["z"])
                ],
            }
            for _, edge in bundled_edges.iterrows()
        ]

        with open(nodes_path, "w") as f:
            json.dump(nodes_json, f)
        with open(edges_path, "w") as f:
            json.dump(edges_json, f)


# Usage
processor = OptimizedGraphProcessor("path_to_your_input_file.graphml")

# Without hammer bundle
g, bundled_edges_3d = processor.process_graph(use_hammer_bundle=False)

# With hammer bundle
g_hammer, bundled_edges_hammer_3d = processor.process_graph(use_hammer_bundle=True)

# Compare results
OptimizedGraphProcessor.save_to_json(
    g, bundled_edges_3d, "nodes_3d.json", "edges_3d.json"
)
OptimizedGraphProcessor.save_to_json(
    g, bundled_edges_hammer_3d, "nodes_hammer_3d.json", "edges_hammer_3d.json"
)
