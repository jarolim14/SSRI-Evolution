import leidenalg as la
import pandas as pd
import community


class PartitionCreator:
    def __init__(self, graph, dataframe):
        """Initialize the PartitionCreator with a graph and dataframe."""
        self.graph = graph
        self.dataframe = dataframe
        self.cluster_sizes = None
        self.time_series_dataframe = None

    def create_partition(self, algorithm="louvain"):
        """Create a partition of the graph using the specified algorithm."""
        if algorithm == "louvain":
            partition = community.best_partition(self.graph)
        elif algorithm == "label_propagation":
            partition = community.label_propagation_communities(self.graph)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        # Add partition information to the dataframe
        self.dataframe["cluster"] = self.dataframe["eid"].map(partition)
        
        # Calculate cluster sizes
        self.cluster_sizes = self.dataframe["cluster"].value_counts()
        
        return partition

    def create_time_series_dataframe(self):
        """Create a time series dataframe from the partition data."""
        # Create a time series dataframe
        time_series_dataframe = (
            self.dataframe.groupby("year")["cluster"].value_counts(normalize=True).unstack()
        )
        
        # Fill NaNs with 0
        time_series_dataframe.fillna(0, inplace=True)
        
        # Rename columns to cluster_nr
        time_series_dataframe.columns = [
            f"cluster_{i}" for i in self.cluster_sizes.index
        ]
        
        # Reset index to make year a column
        time_series_dataframe.reset_index(inplace=True)
        
        self.time_series_dataframe = time_series_dataframe
        return time_series_dataframe

    def create_partition_from_modularityvertexpartition(
        self,
        n_iterations,
        max_comm_size=9999,
        verbose=False,
    ):
        self.partition = la.find_partition(
            self.graph,
            la.ModularityVertexPartition,
            weights="weight",
            n_iterations=n_iterations,
            max_comm_size=max_comm_size,
            seed=1887,
        )
        # assign cluster labels to G
        self.graph.vs["cluster"] = self.partition.membership
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
            self.graph,
            la.CPMVertexPartition,
            resolution_parameter=resolution_parameter,
            weights="weight",
            n_iterations=n_iterations,
            seed=1887,
        )
        # assign cluster labels to G
        self.graph.vs["cluster"] = self.partition.membership
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
        vertices_df = pd.DataFrame([vertex.attributes() for vertex in self.graph.vs])

        # Add the 'cluster' column to the DataFrame based on 'eid'
        self.dataframe = self.dataframe.merge(
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
            Gsub = self.graph.subgraph(
                [v.index for v in self.graph.vs if v["cluster"] == cluster_nr]
            )
            # Use igraph to get eigenvector centrality for the subgraph
            eigenvector_cluster_centralities = Gsub.eigenvector_centrality(
                weights="weight"
            )
            eids = Gsub.vs["eid"]
            for eid, ev in zip(eids, eigenvector_cluster_centralities):
                eid_eigenvector[eid] = ev

        self.dataframe[column_name] = self.dataframe["eid"].map(eid_eigenvector)

        self.graph.vs[column_name] = [eid_eigenvector[eid] for eid in self.graph.vs["eid"]]

    def cluster_title_printer(
        self,
        cluster_nr,
        random=False,
        n=15,
        cols_to_print=["title", "year"],
    ):
        cluster = self.dataframe[self.dataframe["cluster"] == cluster_nr]
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


# example usage
# pc = PartionCreation(G, df)
# pc.create_partition_from_modularityvertexpartition(n_iterations=20)
# pc.add_cluster_to_df()
# pc.calculate_centrality_within_cluster()
# pc.cluster_title_printer(cluster_nr=0, random=True, n=5)
# df_time_series = pc.create_time_series_dataframe()
# df_time_series.head()
