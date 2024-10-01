import json
import pandas as pd
from typing import Dict, List, Optional, Union


class nodesSaver:
    """
    A utility class for saving node data from a DataFrame to JSON format, particularly for use in JavaScript applications.
    """

    @staticmethod
    def save_dataframe_nodes_to_json(
        df: pd.DataFrame,
        paths: Union[str, List[str]],
        return_json: bool = False,
        attributes: List[str] = None,
    ) -> Optional[List[Dict]]:
        """
        Save the DataFrame nodes to one or more JSON files.

        Args:
            df (pd.DataFrame): The input DataFrame containing node data.
            paths (Union[str, List[str]]): Path or list of paths to save the JSON file(s).
            return_json (bool): If True, return the JSON data as well as saving it.
            attributes (List[str]): List of node attributes to include in the JSON.

        Returns:
            Optional[List[Dict]]: List of node dictionaries if return_json is True, else None.

        Raises:
            ValueError: If a specified attribute is missing from the DataFrame.
        """
        if attributes is None:
            attributes = [
                "node_id",
                "node_name",
                "doi",
                "year",
                "title",
                "cluster",
                "centrality",
                "x",
                "y",
                "z",
            ]

        # Check if all attributes are present in the DataFrame
        missing_attributes = [attr for attr in attributes if attr not in df.columns]
        if missing_attributes:
            raise ValueError(f"Missing attributes in DataFrame: {missing_attributes}")

        # Fix encoding of titles
        df["title"] = df["title"].apply(nodesSaver.fix_encoding)

        # Convert DataFrame to list of dictionaries
        nodes_json = df[attributes].to_dict(orient="records")

        # Convert single path to list for consistent processing
        if isinstance(paths, str):
            paths = [paths]

        # Save to all specified paths
        for path in paths:
            with open(path, "w") as f:
                json.dump(nodes_json, f)
            print(f"Graph nodes saved to {path}")

        return nodes_json if return_json else None

    @staticmethod
    def fix_encoding(title: str) -> str:
        """
        Fix the encoding of a string.

        Args:
            title (str): The input string to fix.

        Returns:
            str: The fixed string.
        """
        try:
            decoded_title = title.encode("utf-8").decode("unicode_escape")
            return decoded_title.encode("latin1").decode("utf-8")
        except UnicodeEncodeError:
            # If the above method fails, return the original title
            return title


class edgesSaver:
    def __init__(self, edges_df, nodes_df):
        self.edges_df = edges_df
        self.nodes_df = nodes_df

    def add_color_attr(self):
        """
        Add cluster information to edges based on node clusters.
        If nodes are in the same cluster, the cluster ID is added; otherwise, -1 is added.
        """
        self.edges_df["color"] = [
            (
                self.nodes_df.loc[source, "cluster"]
                if self.nodes_df.loc[source, "cluster"]
                == self.nodes_df.loc[target, "cluster"]
                else -1
            )
            for source, target in zip(self.edges_df["source"], self.edges_df["target"])
        ]

        print(
            f"Color attribute added to edges. \n-1 if inter-clusters, cluster number if intra-cluster edge."
        )
        return self.edges_df

    def transform_edges(
        self, x_col="x", y_col="y", z_col="z", extra_edge_attributes=None
    ):
        """
        Transform edge data from a DataFrame into a list of dictionaries with points and extra attributes.

        Args:
        x_col (str): Name of the column containing x-coordinates. Default is "x".
        y_col (str): Name of the column containing y-coordinates. Default is "y".
        z_col (str): Name of the column containing z-coordinates. Default is "z".
        extra_edge_attributes (list): List of additional attribute names to include. Default is None.

        Returns:
        list: A list of dictionaries, each representing an edge with its points and attributes.
        """
        if extra_edge_attributes is None:
            extra_edge_attributes = []

        def create_edge_object(edge):
            return {
                **{attr: edge[attr] for attr in extra_edge_attributes if attr in edge},
                "points": [
                    {"x": float(x), "y": float(y), "z": float(z)}
                    for x, y, z in zip(edge[x_col], edge[y_col], edge[z_col])
                    if not (pd.isna(x) or pd.isna(y) or pd.isna(z))
                ],
            }

        return [
            create_edge_object(edge) for edge in self.edges_df.to_dict(orient="records")
        ]

    def save_edges_to_json(self, edges_list, output_dir):
        """
        Save the edges list to a JSON file.

        Args:
        edges_list (list): The list of edges to save.
        output_dir (str): The directory path to save the JSON file.
        """
        with open(output_dir, "w") as f:
            json.dump(edges_list, f)
        print(f"Edges data saved to {output_dir}")
