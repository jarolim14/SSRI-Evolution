import numpy as np
import pandas as pd
import networkx as nx


class ZCoordinateAdder:
    """
    A class for adding a z-coordinate to the nodes of a graph based on their centrality values.
    The z-coordinate range is determined by a percentage of the x-y dimension range.
    """

    def __init__(self, g, percentage=50):
        self.g = g
        self.percentage = percentage / 100  # Convert percentage to decimal

    def add_z_coordinate_to_nodes(self):
        """
        Add a z-coordinate to the nodes of the graph based on their centrality values.
        Args:
            g (nx.Graph): The input graph.
            percentage (float): The percentage of x-y dimension range to use for z-coordinate range.
        Returns:
            nx.Graph: The graph with the z-coordinate added to the nodes.
        """
        print(
            f"Z Coordinates are added, based on their centrality values\nAs {self.percentage} percent of the range of the x-y dimension."
        )
        # Calculate the bounds of x and y coordinates
        xvalues = [attributes["x"] for _, attributes in self.g.nodes(data=True)]
        yvalues = [attributes["y"] for _, attributes in self.g.nodes(data=True)]
        min_x, max_x = min(xvalues), max(xvalues)
        min_y, max_y = min(yvalues), max(yvalues)

        # Calculate the range of x and y
        x_range = max_x - min_x
        y_range = max_y - min_y

        # Calculate the maximum z range based on the larger of x or y range
        max_z_range = max(x_range, y_range) * self.percentage

        # Extract centrality values from nodes
        centralities = np.array(
            [self.g.nodes[node]["centrality"] for node in self.g.nodes]
        )

        # Normalize centrality values to range [0, 1]
        centrality_min = centralities.min()
        centrality_max = centralities.max()
        centralities_normalized = (centralities - centrality_min) / (
            centrality_max - centrality_min
        )

        # Scale the normalized centralities to the desired z range
        z_coordinates = centralities_normalized * max_z_range

        # Add z-coordinate to nodes
        for i, node in enumerate(self.g.nodes):
            self.g.nodes[node]["z"] = z_coordinates[i]

        print("Bounds of the layout:")
        print(f"Min x: {min_x}, Max x: {max_x}")
        print(f"Min y: {min_y}, Max y: {max_y}")
        print(f"Min z: {z_coordinates.min()}, Max z: {z_coordinates.max()}")
        print("Z coordinate added to nodes")

        return self.g
