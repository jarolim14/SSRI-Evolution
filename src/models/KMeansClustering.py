import os

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


class KMeansClustering:
    def __init__(self, df, num_clusters, embeddings_column="embeddings"):
        """
        Initializes the KMeansClustering class.

        Parameters:
        df (pd.DataFrame): DataFrame containing the embeddings.
        num_clusters (int): Number of clusters for KMeans.
        embeddings_column (str): Column name where embeddings are stored. Default is "embeddings".
        """
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input 'df' should be a pandas DataFrame")

        self.df = df.copy()
        self.df.reset_index(drop=True, inplace=True)
        self.num_clusters = num_clusters
        self.embeddings_column = embeddings_column

    def cluster_embeddings(self, text_embeddings):
        """
        Performs KMeans clustering on text embeddings.

        Parameters:
        text_embeddings (np.array): Array of text embeddings.

        Returns:
        KMeans object after fitting.
        """
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=1887).fit(
            text_embeddings
        )
        return kmeans

    def kmeans_clustering(self, save_path=None):
        """
        Applies KMeans clustering to the DataFrame. Creates columns for cluster labels and distances to cluster centers. Optionally saves the result.

        Parameters:
        save_path (str): Path to save the DataFrame. If None, the DataFrame is not saved. Default is None.
        """
        if self.embeddings_column not in self.df.columns:
            raise ValueError(
                f"Column '{self.embeddings_column}' not found in DataFrame"
            )

        text_embeddings = self.df[self.embeddings_column].tolist()
        kmeans = self.cluster_embeddings(text_embeddings)
        self.df["cluster"] = kmeans.labels_
        self.df["distance_to_centroid"] = kmeans.transform(text_embeddings).min(axis=1)

        # Calculate and print silhouette score
        silhouette_avg = silhouette_score(text_embeddings, kmeans.labels_)
        print(f"Silhouette Score: {silhouette_avg:.2f}")

        distances = kmeans.transform(text_embeddings)
        df_distances = pd.DataFrame(
            distances, columns=[f"distance_to_{i}" for i in range(self.num_clusters)]
        )
        self.df = pd.concat([self.df, df_distances], axis=1)

        if save_path:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            self.df.to_pickle(save_path)
            print(f"DataFrame saved to {save_path}")

    def top_n_per_cluster(self, n=10, save_path=None, additional_cols=None):
        """
        Retrieves the top n entries per cluster based on the distance to the centroid.

        Parameters:
        n (int): Number of top entries to retrieve per cluster. Default is 10.
        save_path (str): Path to save the Excel file. Required if save is True.
        additional_cols (list): List of additional column names to include in the saved file.
        """
        if (
            "cluster" not in self.df.columns
            or "distance_to_centroid" not in self.df.columns
        ):
            raise ValueError(
                "Required columns 'cluster' or 'distance_to_centroid' not found in DataFrame"
            )

        top_n = self.df.groupby("cluster").apply(
            lambda x: x.nsmallest(n, "distance_to_centroid")
        )

        if save_path:
            if not additional_cols:
                additional_cols = []

            distance_cols = [
                col for col in self.df.columns if col.startswith("distance")
            ]
            columns_to_save = additional_cols + distance_cols
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))

            top_n[columns_to_save].round(2).to_excel(save_path)
            print(f"Top {n} per cluster saved to {save_path}")

        return top_n


# Example usage
# kmeans_cluster = KMeansClustering(df, 5)  # Assuming df is your DataFrame and 5 is the number of clusters
# kmeans_cluster.kmeans_clustering("/path/to/save/your_dataframe.pkl")
# top_n_clusters = kmeans_cluster.top_n_per_cluster(n=10, save=True, save_path="/path/to/save/top_n_clusters.xlsx", additional_cols=["title", "year"])
