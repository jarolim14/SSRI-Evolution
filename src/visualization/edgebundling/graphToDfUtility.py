import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from typing import List, Tuple, Union, Optional


class graphToDfUtility:
    """
    A utility class for converting igraph graph attributes to pandas DataFrames
    and performing DataFrame operations related to graph data.
    """

    @staticmethod
    def minmax_normalize(
        data: Union[List[float], np.ndarray], new_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        data = np.array(data)
        if len(data) == 0:
            raise ValueError("Input data is empty")

        data_min, data_max = np.min(data), np.max(data)
        if data_min == data_max:
            raise ValueError("All values in the input data are identical")

        normalized_data = (data - data_min) / (data_max - data_min)

        if new_range != (0, 1):
            new_min, new_max = new_range
            normalized_data = normalized_data * (new_max - new_min) + new_min

        return normalized_data

    def minmax_denormalize(
        normalized_data: Union[List[float], np.ndarray],
        original_range: Tuple[float, float],
        current_range: Tuple[float, float] = (0, 1),
    ) -> np.ndarray:

        normalized_data = np.array(normalized_data)
        if len(normalized_data) == 0:
            raise ValueError("Input data is empty")

        orig_min, orig_max = original_range
        curr_min, curr_max = current_range

        if orig_min >= orig_max or curr_min >= curr_max:
            raise ValueError("Invalid range: min should be less than max")

        # First, normalize to [0, 1] if not already
        if current_range != (0, 1):
            normalized_data = (normalized_data - curr_min) / (curr_max - curr_min)

        # Then, scale to original range
        denormalized_data = normalized_data * (orig_max - orig_min) + orig_min

        return denormalized_data

    @staticmethod
    def edges_to_dataframe(g: ig.Graph) -> pd.DataFrame:
        """
        Convert the edges and their attributes of an igraph graph to a pandas DataFrame.

        Args:
            g (ig.Graph): The input igraph graph.

        Returns:
            pd.DataFrame: A DataFrame containing edge attributes, source, and target nodes.
        """
        # check if networkx or igraph
        if isinstance(g, nx.Graph):
            g = ig.Graph.from_networkx(g)
        # Extract edge attributes
        edge_data = {attr: g.es[attr] for attr in g.es.attributes()}

        # Add source and target node indices
        edge_data["source"] = [e.source for e in g.es]
        edge_data["target"] = [e.target for e in g.es]

        # Convert to DataFrame
        edge_dataframe = pd.DataFrame(edge_data)

        return edge_dataframe

    @staticmethod
    def nodes_to_dataframe(
        g: ig.Graph,
        normalize_coordinates: bool = False,
        drop_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert the nodes and their attributes of an igraph graph to a pandas DataFrame.

        Args:
            g (ig.Graph): The input igraph graph.
            drop_columns (Optional[List[str]]): A list of column names to drop from the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame containing node attributes and node indices.
        """
        # check if networkx or igraph
        if isinstance(g, nx.Graph):
            g = ig.Graph.from_networkx(g)
        # Extract node attributes
        node_data = {attr: g.vs[attr] for attr in g.vs.attributes()}

        # Add node indices
        node_dataframe = pd.DataFrame(node_data)
        node_dataframe["node_index"] = [n.index for n in g.vs]
        if normalize_coordinates:
            # Calculate min/max for coordinates
            xmin, xmax = np.min(node_dataframe["x"]), np.max(node_dataframe["x"])
            ymin, ymax = np.min(node_dataframe["y"]), np.max(node_dataframe["y"])
            zmin, zmax = np.min(node_dataframe["z"]), np.max(node_dataframe["z"])

            # Normalize coordinates
            node_dataframe["x"] = minmax_normalize(node_dataframe["x"], xmin, xmax)
            node_dataframe["y"] = minmax_normalize(node_dataframe["y"], ymin, ymax)
            node_dataframe["z"] = minmax_normalize(node_dataframe["z"], zmin, zmax)
            print("Coordinates normalized to [0, 1] range.")

        # Drop specified columns if provided
        if drop_columns:
            node_dataframe = node_dataframe.drop(columns=drop_columns, errors="ignore")

        return node_dataframe

    @staticmethod
    def create_edge_df_with_source_target_coords(g: ig.Graph) -> pd.DataFrame:
        """
        Create a DataFrame containing edge information with source and target coordinates.

        This method processes an igraph Graph object and extracts edge information,
        including weights, IDs, source and target nodes, and their coordinates.
        It also calculates the length of each edge segment.

        Parameters:
        -----------
        g : ig.Graph
            The input graph object from which to extract edge information.

        Returns:
        --------
        pd.DataFrame
            A DataFrame containing the following columns:
            - weight: Edge weight
            - edge_id: Unique identifier for each edge
            - source: Source node ID
            - target: Target node ID
            - source_x, source_y, source_z: Coordinates of the source node
            - target_x, target_y, target_z: Coordinates of the target node
            - segment_length: Euclidean distance between source and target nodes

        Notes:
        ------
        - Assumes the graph nodes have 'x', 'y', and 'z' attributes for coordinates.
        - Prints statistics about segment lengths after creating the DataFrame.
        """
        edge_data = []
        for edge in g.es:
            source = edge.source
            target = edge.target
            source_coords = (g.vs[source]["x"], g.vs[source]["y"], g.vs[source]["z"])
            target_coords = (g.vs[target]["x"], g.vs[target]["y"], g.vs[target]["z"])

            edge_data.append(
                {
                    "weight": edge["weight"],
                    "edge_id": edge["edge_id"],
                    "source": source,
                    "target": target,
                    "source_x": source_coords[0],
                    "source_y": source_coords[1],
                    "source_z": source_coords[2],
                    "target_x": target_coords[0],
                    "target_y": target_coords[1],
                    "target_z": target_coords[2],
                    "segment_length": graphToDfUtility.distance_between(
                        source_coords, target_coords
                    ),
                }
            )

        df = pd.DataFrame(edge_data)

        # Print segment length statistics
        print("Segment length statistics:")
        print(f"Min: {df['segment_length'].min():.2f}")
        print(f"Max: {df['segment_length'].max():.2f}")
        print(f"Mean: {df['segment_length'].mean():.2f}")
        print(f"Median: {df['segment_length'].median():.2f}")

        return df

    @staticmethod
    def distance_between(
        point1: Tuple[float, float, float], point2: Tuple[float, float, float]
    ) -> float:
        """Calculate the Euclidean distance between two 3D points."""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
