import leidenalg as la
import pandas as pd


class PartitionCreator:
    def __init__(self, G, df):
        self.G = G
        self.df = df

    def create_partition_from_modularityvertexpartition(
        self,
        n_iterations,
        max_comm_size=9999,
        verbose=False,
    ):
        self.partition = la.find_partition(
            self.G,
            la.ModularityVertexPartition,
            weights="weight",
            n_iterations=n_iterations,
            max_comm_size=max_comm_size,
            seed=1887,
        )
        # assign cluster labels to G
        self.G.vs["cluster"] = self.partition.membership
        # sizes of clusters
        self.cluster_sizes = pd.Series(self.partition.sizes()).sort_values(
            ascending=False
        )
        self.modularity = self.partition.modularity
        if verbose:
            print(f"Number of clusters: {len(self.cluster_sizes)}")
            # print(f"Quality: {self.partition.quality()}")
            print(f"Modularity: { self.modularity}")

    def create_partition_from_cmpvertexpartition(
        self,
        n_iterations=20,
        resolution_parameter=0.002,
        verbose=False,
        cluster_column_name=None,
        centrality_column_name=None,
    ):
        self.partition = la.find_partition(
            self.G,
            la.CPMVertexPartition,
            resolution_parameter=resolution_parameter,
            weights="weight",
            n_iterations=n_iterations,
            seed=1887,
        )
        # assign cluster labels to G
        self.G.vs["cluster"] = self.partition.membership
        # sizes of clusters
        self.cluster_sizes = pd.Series(self.partition.sizes()).sort_values(
            ascending=False
        )

        if cluster_column_name:
            self.add_cluster_to_df(column_name=cluster_column_name)
        if centrality_column_name:
            self.add_centrality_to_df_and_graph(column_name=centrality_column_name)
        if verbose:
            print(f"Number of clusters: {len(self.cluster_sizes)}")
            print(f"Quality: {self.partition.quality()}")

    def add_cluster_to_df(self, column_name="cluster"):
        # Extract vertex attributes as a list of dictionaries
        vertices_df = pd.DataFrame([vertex.attributes() for vertex in self.G.vs])

        # Add the 'cluster' column to the DataFrame based on 'eid'
        self.df = self.df.merge(
            vertices_df[["eid", "cluster"]], on="eid", how="left"
        ).rename(columns={"cluster": column_name})

    def add_centrality_to_df_and_graph(
        self, column_name="eigenvector_cluster_centrality"
    ):
        """
        Calculate a specified metric for each paper within its cluster and add it as a new column in the DataFrame.

        Args:
            metric_name (str): The name of the metric to calculate. Supported metrics: 'degree', 'closeness', 'betweenness', 'eigenvector', 'pagerank'.
        """

        eid_eigenvector = {}

        for cluster_nr in self.cluster_sizes.index:
            # Extract the subgraph for the current cluster
            Gsub = self.G.subgraph(
                [v.index for v in self.G.vs if v["cluster"] == cluster_nr]
            )
            # Use igraph to get eigenvector centrality for the subgraph
            eigenvector_cluster_centralities = Gsub.eigenvector_centrality(
                weights="weight"
            )
            eids = Gsub.vs["eid"]
            for eid, ev in zip(eids, eigenvector_cluster_centralities):
                eid_eigenvector[eid] = ev

        self.df[column_name] = self.df["eid"].map(eid_eigenvector)

        self.G.vs[column_name] = [eid_eigenvector[eid] for eid in self.G.vs["eid"]]

    def cluster_title_printer(
        self,
        cluster_nr,
        random=False,
        n=15,
        cols_to_print=["title", "year"],
    ):
        cluster = self.df[self.df["cluster"] == cluster_nr]
        cluster = cluster.sort_values(
            by="eigenvector_cluster_centrality", ascending=False
        )
        print(f"Cluster {cluster_nr} has {len(cluster)} papers")
        print(
            f"Mean and SD of year is {cluster['year'].mean():.1f} +/- {cluster['year'].std():.1f}"
        )
        if len(cluster) < n:
            n = len(cluster)

        if random:
            cluster = cluster.sample(n)

        for _, row in cluster.head(n).iterrows():
            print("")
            [print(f"{col}: {row[col]}") for col in cols_to_print]

    def create_time_series_df(self):
        # create a time series df
        df_time_series = (
            self.df.groupby("year")["cluster"].value_counts(normalize=True).unstack()
        )
        # fill NaNs with 0
        df_time_series.fillna(0, inplace=True)
        # columns to cluser_nr
        df_time_series.columns = [
            "cluster" + "_" + str(i) for i in self.cluster_sizes.index
        ]
        # index to column
        df_time_series.reset_index(inplace=True)
        self.df_time_series = df_time_series
        return df_time_series


# example usage
# pc = PartionCreation(G, df)
# pc.create_partition_from_modularityvertexpartition(n_iterations=20)
# pc.add_cluster_to_df()
# pc.calculate_centrality_within_cluster()
# pc.cluster_title_printer(cluster_nr=0, random=True, n=5)
# df_time_series = pc.create_time_series_df()
# df_time_series.head()
