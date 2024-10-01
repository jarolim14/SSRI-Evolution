import pandas as pd


class postBundlingUtility:
    """
    A class to process edge data for graph visualization.

    This class handles the creation of bundled and straight edges from input dataframes,
    concatenates the results, and cleans the edge attributes for further analysis or visualization.

    Attributes:
    bundled_edge_pts (pd.DataFrame): DataFrame containing points for bundled edges.
    edges_to_bundle_df (pd.DataFrame): DataFrame mapping edges to their source and target nodes.
    edge_df_with_source_target_coords (pd.DataFrame): DataFrame containing all edge coordinates and their lengths.
    threshold (float): Length threshold to determine which edges are considered straight.
    """

    def __init__(
        self,
        bundled_edge_pts,
        edges_to_bundle_df,
        edge_df_with_source_target_coords,
        threshold,
    ):
        self.bundled_edge_pts = bundled_edge_pts
        self.edges_to_bundle_df = edges_to_bundle_df
        self.edge_df_with_source_target_coords = edge_df_with_source_target_coords
        self.threshold = threshold

    def create_bundled_edges_df(self):
        sub_bundle_idx = 0
        x, y, z = [], [], []
        bundled_edges_dict = {}

        for i in range(len(self.bundled_edge_pts)):
            # if not nan
            if pd.isna(self.bundled_edge_pts.iloc[i, 0]):
                target = self.edges_to_bundle_df.loc[sub_bundle_idx, "target"]
                source = self.edges_to_bundle_df.loc[sub_bundle_idx, "source"]
                bundled_edges_dict[(source, target)] = {
                    "x": x,
                    "y": y,
                    "z": z,
                }
                sub_bundle_idx += 1
                x, y, z = [], [], []
            else:
                x.append(self.bundled_edge_pts.iloc[i, 0])
                y.append(self.bundled_edge_pts.iloc[i, 1])
                z.append(self.bundled_edge_pts.iloc[i, 2])

        # Create DataFrame
        bundled_edges_df = pd.DataFrame.from_dict(bundled_edges_dict, orient="index")
        bundled_edges_df["source"] = [x[0] for x in bundled_edges_df.index]
        bundled_edges_df["target"] = [x[1] for x in bundled_edges_df.index]
        bundled_edges_df.reset_index(drop=True, inplace=True)
        print(f"Number of bundled edges: {bundled_edges_df.shape[0]}")
        return bundled_edges_df

    def create_straight_edges_df(self):
        straight_edges_mask = (
            self.edge_df_with_source_target_coords["segment_length"] <= self.threshold
        )

        straight_edges_df = self.edge_df_with_source_target_coords.loc[
            straight_edges_mask,
            [
                "source",
                "target",
                "source_x",
                "source_y",
                "source_z",
                "target_x",
                "target_y",
                "target_z",
            ],
        ].reset_index(drop=True)

        straight_edges_df["x"] = [
            [source_x, target_x]
            for source_x, target_x in zip(
                straight_edges_df["source_x"], straight_edges_df["target_x"]
            )
        ]
        straight_edges_df["y"] = [
            [source_y, target_y]
            for source_y, target_y in zip(
                straight_edges_df["source_y"], straight_edges_df["target_y"]
            )
        ]
        straight_edges_df["z"] = [
            [source_z, target_z]
            for source_z, target_z in zip(
                straight_edges_df["source_z"], straight_edges_df["target_z"]
            )
        ]

        # Drop coordinate columns
        straight_edges_df = straight_edges_df.drop(
            columns=[
                "source_x",
                "source_y",
                "source_z",
                "target_x",
                "target_y",
                "target_z",
            ]
        )

        return straight_edges_df

    def concat_and_clean_edges(self, bundled_edges_df, straight_edges_df):
        # Concatenate bundled and straight edges
        final_edges_df = pd.concat(
            [bundled_edges_df, straight_edges_df], ignore_index=True
        )

        # Round to 10 decimal places in x, y, z
        for coord in ["x", "y", "z"]:
            final_edges_df[coord] = final_edges_df[coord].apply(
                lambda points: [round(i, 10) for i in points]
            )

        return final_edges_df

    def post_process_edges(self):
        bundled_edges_df = self.create_bundled_edges_df()
        straight_edges_df = self.create_straight_edges_df()
        final_edges_df = self.concat_and_clean_edges(
            bundled_edges_df, straight_edges_df
        )

        print(f"Number of total edges: {final_edges_df.shape[0]}")
        print(f"Number of straight edges: {straight_edges_df.shape[0]}")

        return final_edges_df
