import numpy as np
import pandas as pd
import igraph as ig
import networkx as nx
from typing import List, Tuple, Union, Optional


class GraphProcessingUtility:
    """
    A utility class for converting igraph and NetworkX graphs to pandas DataFrames,
    performing min-max normalization, and adding coordinate and distance data to edges.
    """

    @staticmethod
    def minmax_normalize(
        data: Union[List[float], np.ndarray], new_range: Tuple[float, float] = (0, 1)
    ) -> np.ndarray:
        """
        Normalize the input data to a specified range (default is [0, 1]).

        Args:
            data (Union[List[float], np.ndarray]): The data to be normalized.
            new_range (Tuple[float, float]): The desired output range (default is (0, 1)).

        Returns:
            np.ndarray: The normalized data.
        """
        data = np.array(data)
        if len(data) == 0:
            raise ValueError("Input data is empty")

        data_min, data_max = np.min(data), np.max(data)
        if data_min == data_max:
            raise ValueError("All values in the input data are identical")

        normalized_data = (data - data_min) / (data_max - data_min)
        new_min, new_max = new_range
        return normalized_data * (new_max - new_min) + new_min

    @staticmethod
    def edges_to_dataframe(g: Union[ig.Graph, nx.Graph]) -> pd.DataFrame:
        """
        Convert the edges and their attributes of a graph (igraph or NetworkX) to a pandas DataFrame.

        Args:
            g (Union[ig.Graph, nx.Graph]): The input graph.

        Returns:
            pd.DataFrame: A DataFrame containing edge attributes, source, and target nodes.
        """
        if isinstance(g, nx.Graph):
            g = ig.Graph.from_networkx(g)

        edge_data = {attr: g.es[attr] for attr in g.es.attributes()}
        edge_data["source"] = [g.vs[e.source]["node_index"] for e in g.es]
        edge_data["target"] = [g.vs[e.target]["node_index"] for e in g.es]

        return pd.DataFrame(edge_data)

    @staticmethod
    def nodes_to_dataframe(
        g: Union[ig.Graph, nx.Graph],
        normalize_coordinates: bool = False,
        drop_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Convert the nodes and their attributes of a graph (igraph or NetworkX) to a pandas DataFrame.

        Args:
            g (Union[ig.Graph, nx.Graph]): The input graph.
            normalize_coordinates (bool): Whether to normalize coordinates (default is False).
            drop_columns (Optional[List[str]]): List of column names to drop.

        Returns:
            pd.DataFrame: A DataFrame containing node attributes and indices.
        """
        if isinstance(g, nx.Graph):
            g = ig.Graph.from_networkx(g)

        node_data = {attr: g.vs[attr] for attr in g.vs.attributes()}
        node_dataframe = pd.DataFrame(node_data)

        if normalize_coordinates:
            # Normalize x, y, z coordinates
            for coord in ["x", "y", "z"]:
                node_dataframe[coord] = GraphProcessingUtility.minmax_normalize(
                    node_dataframe[coord]
                )
            print("Coordinates normalized to [0, 1] range.")

        if drop_columns:
            node_dataframe = node_dataframe.drop(columns=drop_columns, errors="ignore")

        return node_dataframe

    @staticmethod
    def create_edge_df_with_source_target_coords(
        edges_df: pd.DataFrame, nodes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Adds the source and target node coordinates (x, y, z) to the edges DataFrame.

        Args:
            edges_df (pd.DataFrame): DataFrame containing edges with 'source' and 'target' columns.
            nodes_df (pd.DataFrame): DataFrame containing node data with 'node_index', 'x', 'y', 'z' columns.

        Returns:
            pd.DataFrame: edges_df with added 'source_x', 'source_y', 'source_z', 'target_x', 'target_y', 'target_z' columns.
        """

        def add_node_coords(
            edges_df: pd.DataFrame,
            nodes_df: pd.DataFrame,
            node_column: str,
            prefix: str,
        ) -> pd.DataFrame:
            """
            Helper function to merge node coordinates (x, y, z) to edges_df for a given node column (source or target).

            Args:
                edges_df (pd.DataFrame): DataFrame containing edges.
                nodes_df (pd.DataFrame): DataFrame containing nodes and their coordinates.
                node_column (str): Column name in edges_df ('source' or 'target').
                prefix (str): Prefix for the new coordinate columns ('source' or 'target').

            Returns:
                pd.DataFrame: edges_df with merged coordinates for the specified node_column.
            """
            return (
                edges_df.merge(
                    nodes_df[
                        ["node_index", "x", "y", "z"]
                    ],  # Only select relevant columns
                    how="left",  # Left join to preserve all rows from edges_df
                    left_on=node_column,  # Match on the specified column in edges_df ('source' or 'target')
                    right_on="node_index",  # Match on 'node_index' in nodes_df
                )
                .rename(
                    columns={  # Rename the merged columns with the given prefix
                        "x": f"{prefix}_x",
                        "y": f"{prefix}_y",
                        "z": f"{prefix}_z",
                    }
                )
                .drop(
                    columns=["node_index"]
                )  # Drop 'node_index' as it is no longer needed
            )

        edges_with_coords = add_node_coords(edges_df, nodes_df, "source", "source")
        edges_with_coords = add_node_coords(
            edges_with_coords, nodes_df, "target", "target"
        )

        return edges_with_coords

    @staticmethod
    def add_segment_length_to_edge_df(edges_with_coords: pd.DataFrame) -> pd.DataFrame:
        """
        Add the segment length (Euclidean distance between source and target nodes) to the edges DataFrame.

        Args:
            edges_with_coords (pd.DataFrame): DataFrame containing edges with source and target coordinates.

        Returns:
            pd.DataFrame: edges_with_coords with an added 'segment_length' column.
        """
        edges_with_coords["segment_length"] = edges_with_coords.apply(
            lambda row: GraphProcessingUtility.distance_between(
                (row["source_x"], row["source_y"], row["source_z"]),
                (row["target_x"], row["target_y"], row["target_z"]),
            ),
            axis=1,
        )

        # Print statistics
        print("Segment length statistics:")
        print(f"Min: {edges_with_coords['segment_length'].min():.2f}")
        print(f"Max: {edges_with_coords['segment_length'].max():.2f}")
        print(f"Mean: {edges_with_coords['segment_length'].mean():.2f}")
        print(f"Median: {edges_with_coords['segment_length'].median():.2f}")

        return edges_with_coords

    @staticmethod
    def distance_between(
        point1: Tuple[float, float, float], point2: Tuple[float, float, float]
    ) -> float:
        """Calculate the Euclidean distance between two 3D points."""
        return np.sqrt(sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)))
