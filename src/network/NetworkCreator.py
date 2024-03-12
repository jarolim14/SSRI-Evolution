import itertools

import networkx as nx
import numpy as np
from sklearn.neighbors import NearestNeighbors


class WeightedNetworkCreator:
    """
    A class used to create a weighted network from a given dataframe.

    ...

    Attributes
    ----------
    df : DataFrame
        a pandas DataFrame containing the data
    alpha : float
        a weighting factor for the link matrix

    Methods
    -------
    get_nearest_neighbours(k=15):
        Returns the similarities and indices of the k nearest neighbours for each data point.
    link_matrix_knn(similarities, indices):
        Creates a K-Nearest Neighbors link matrix based on the similarities and indices.
    link_matrix_citations(knn_matrix):
        Enhances the KNN matrix with citation links.
    make_matrix_symmetric(matrix):
        Makes a given matrix symmetric by averaging the upper and lower triangular parts.
    create_network(matrix):
        Creates a weighted undirected network from the given matrix.
    add_data_to_nodes(G, col_list=["eid", "title", "year", "abstract", "doi", "unique_auth_year"]):
        Adds data to the nodes of the given network from the dataframe.
    """

    def __init__(self, df, alpha=0.5):
        print("Initializing WeightedNetworkCreator")
        self.alpha = alpha
        self.df = df.reset_index(drop=True)
        self.info_log = {}

    def get_nearest_neighbours(self, k=15):
        print("Getting nearest neighbours...")
        X = np.array(self.df["specter2_embeddings"].tolist())
        knn = NearestNeighbors(n_neighbors=k, metric="cosine")
        knn.fit(X)
        distances, indices = knn.kneighbors(X)
        similarities = 1 - distances
        return similarities, indices

    def link_matrix_knn(self, similarities, indices):
        print("Creating KNN link matrix...")
        knn_matrix = np.zeros((len(self.df), len(self.df)), dtype=np.float32)
        for i, neighbourhood in enumerate(indices):
            for j, neighbour in enumerate(neighbourhood):
                sim_score = similarities[i, j] * (1 - self.alpha)
                if i != neighbour:
                    knn_matrix[i, neighbour] = sim_score
        return knn_matrix

    def link_matrix_citations(self, knn_matrix):
        print("Creating citation link matrix...")
        edgelist = [
            list(pair)
            for _, row in self.df.iterrows()
            for pair in itertools.product([row["eid"]], row["filtered_reference_eids"])
        ]
        print(f"Number of edges from citations: {len(edgelist)}")
        self.info_log["num_edges_from_citations"] = len(edgelist)
        doc_to_index = dict(zip(self.df["eid"], self.df.index))
        for citing_eid, cited_eid in edgelist:
            citing_idx = doc_to_index[citing_eid]
            cited_idx = doc_to_index[cited_eid]
            knn_matrix[citing_idx, cited_idx] += 1 * self.alpha
        return knn_matrix

    def make_matrix_symmetric(self, matrix):
        """Make a matrix symmetric by averaging the upper and lower triangular parts.
        Only average if both values are non-zero.
        """
        print("Making matrix symmetric...")

        # Vectorized averaging of symmetric elements
        avg_matrix = (matrix + matrix.T) / 2

        # Create masks for zero elements in the original matrix
        mask_i_zero = matrix == 0
        mask_j_zero = matrix.T == 0

        # Apply the masks to get the final averaged matrix
        # If an element in 'matrix' is zero, take the value from 'matrix.T' (and vice versa)
        new_matrix = np.where(
            mask_i_zero, matrix.T, np.where(mask_j_zero, matrix, avg_matrix)
        )

        # Remove redundant edges (keeping the upper triangular part)
        new_matrix = np.triu(new_matrix)
        return new_matrix

    def create_network(self, matrix):
        """
        Create a weighted undirected network the matrix
        """
        print("Creating weighted undirected network...")
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(matrix)
        print(f"Number of nodes: {G.number_of_nodes()}")
        print(f"Number of edges: {G.number_of_edges()}")
        edges_from_citations = self.info_log["num_edges_from_citations"]
        print(
            f"Percentage of edges from citations: {edges_from_citations / G.number_of_edges() * 100:.2f}%"
        )
        edges_from_knn = G.number_of_edges() - edges_from_citations
        print(
            f"Percentage of edges from KNN: {edges_from_knn / G.number_of_edges() * 100:.2f}%"
        )
        self.info_log["num_edges_from_knn"] = edges_from_knn
        self.info_log["num_edges"] = G.number_of_edges()
        self.info_log["num_nodes"] = G.number_of_nodes()
        self.info_log["length_of_df"] = len(self.df)
        return G

    def add_data_to_nodes(
        self,
        G,
        col_list=["eid", "title", "year", "abstract", "doi", "unique_auth_year"],
    ):
        """
        Add data to nodes from df
        """
        print("Adding data to nodes...")
        # Add data to nodes from df
        for node in G.nodes():
            node_data = self.df.loc[node, col_list]  # Fetch all required data at once
            for col in col_list:
                G.nodes[node][col] = node_data[col]
        return G

    def get_info_log(self):
        return self.info_log


class DirectedNetworkCreator:
    """
    A class used to create a citation network from a given DataFrame.

    Attributes
    ----------
    df : DataFrame
        A pandas DataFrame containing the data.

    Methods
    -------
    create_network():
        Creates a directed network from the given DataFrame.
    add_data_to_nodes(G, col_list=None):
        Adds data to the nodes of the given network from the DataFrame.
    """

    def __init__(self, df):
        """
        Initialize CitationNetworkCreator.

        Parameters
        ----------
        df : DataFrame
            A pandas DataFrame containing the data.
        """
        print("Initializing DirectedNetworkCreator")
        self.df = df.reset_index(drop=True)
        self.info_log = {}

    def create_network(self):
        """
        Create a directed network from the DataFrame.

        Returns
        -------
        G : DiGraph
            A directed graph representing the citation network.
        """
        print("Creating directed network...")
        # Create a dictionary to map document IDs to indexes
        doc_to_index = dict(zip(self.df["eid"], self.df.index))

        # Generate the edgelist directly using the dictionary
        edgelist = [
            [doc_to_index[citing_eid], doc_to_index[cited_eid]]
            for _, row in self.df.iterrows()
            for citing_eid, cited_eid in itertools.product(
                [row["eid"]], row["filtered_reference_eids"]
            )
        ]

        G = nx.DiGraph()
        G.add_edges_from(edgelist)

        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        print(f"Number of nodes: {num_nodes}")
        print(f"Number of edges: {num_edges}")

        self.info_log["num_edges"] = num_edges
        self.info_log["num_nodes"] = num_nodes
        self.info_log["length_of_df"] = len(self.df)

        return G

    def add_data_to_nodes(self, G, col_list=None):
        """
        Add data to nodes from the DataFrame.

        Parameters
        ----------
        G : DiGraph
            A directed graph.
        col_list : list of str, optional
            A list of column names to add to the nodes. Default is None, which adds all columns.

        Returns
        -------
        G : DiGraph
            The updated directed graph with node data.
        """
        print("Adding data to nodes...")

        for node in G.nodes():
            for col in col_list:
                G.nodes[node][col] = self.df.loc[node, col]

        return G

    def get_info_log(self):
        """
        Get information log about the created network.

        Returns
        -------
        dict
            A dictionary containing information about the network.
        """
        return self.info_log
