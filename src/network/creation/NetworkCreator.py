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

    def prettify_network(
        self,
        G,
        col_list=["year", "title", "eid"],
    ):
        """
        Add data to nodes from df using numeric IDs.
        """
        G_new = nx.Graph()
        id_mapping = {}

        for i, node in enumerate(G.nodes()):
            # Use numeric index as the node ID
            id_mapping[node] = i
            G_new.add_node(i)

            # Add other attributes from the DataFrame to the node
            node_data = self.df.loc[node, col_list]
            for col in col_list:
                G_new.nodes[i][col] = node_data[col]

        # Add edges, mapping original nodes to new numeric IDs
        for u, v, data in G.edges(data=True):
            G_new.add_edge(id_mapping[u], id_mapping[v], **data)

        return G_new

    def get_info_log(self):
        return self.info_log


class DirectedNetworkCreator:
    def __init__(self, df, data_to_add=["eid", "title", "year"]):
        self.df = df
        self.G = nx.DiGraph()
        self.eid_to_node = {}
        self.data_to_add = data_to_add
        self.info_log = {}

    def build_graph(self):
        self.create_node_mapping()
        self.add_nodes()
        self.add_edges_and_node_attributes()

    def create_node_mapping(self):
        self.eid_to_node = {
            row["eid"]: row["unique_auth_year"] for _, row in self.df.iterrows()
        }

    def add_nodes(self):
        self.G.add_nodes_from(self.df["unique_auth_year"])

    def add_edges_and_node_attributes(self):
        for _, row in self.df.iterrows():
            citing_node = row["unique_auth_year"]
            cited_nodes = [
                self.eid_to_node[cited_eid]
                for cited_eid in row["filtered_reference_eids"]
            ]
            edges = [(citing_node, cited_node) for cited_node in cited_nodes]
            self.G.add_edges_from(edges)

            # Add node attributes
            for col in self.data_to_add:
                self.G.nodes[citing_node][col] = row[col]

    def get_graph_info(self):
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        return f"Number of nodes: {num_nodes}\nNumber of edges: {num_edges}"

    def get_info_log(self):
        num_nodes = self.G.number_of_nodes()
        num_edges = self.G.number_of_edges()
        self.info_log["num_edges"] = num_edges
        self.info_log["num_nodes"] = num_nodes
        nr_isolated_nodes = len(list(nx.isolates(self.G)))
        self.info_log["nr_isolated_nodes"] = nr_isolated_nodes
        return self.info_log


# example usage of directed network creator
# from network.NetworkCreator import DirectedNetworkCreator
#
# # Create a directed network from the dataframe
# dnc = DirectedNetworkCreator(df)
# dnc.build_graph()
# print(dnc.get_graph_info())
# info_log = dnc.get_info_log()
# print(info_log)
