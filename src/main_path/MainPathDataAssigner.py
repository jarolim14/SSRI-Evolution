class MainPathDataAssigner:
    """
    A class to assign data from a pandas DataFrame to nodes of a networkx graph based on a matching column,
    and to modify node attributes within the graph.

    Attributes:
        mp (networkx.Graph): The graph to which data will be added or modified.
        df (pandas.DataFrame): The DataFrame from which data will be sourced.
        attr_to_assign (list): List of attributes to assign from DataFrame to graph nodes.
        cluster_label_dict (dict, optional): Dictionary mapping cluster IDs to their labels.
        cluster_color_dict (dict, optional): Dictionary mapping cluster IDs to color information.

    Methods:
        assign_data_to_mp(match_col="eid"):
            Assigns attributes from the DataFrame to corresponding nodes in the graph.

        remove_data(attr_to_remove=["x", "y", "size", "shape"]):
            Removes specified attributes from all nodes in the graph.

        apply_cluster_labels():
            Applies cluster labels to nodes based on the provided cluster_label_dict.
            
        apply_cluster_colors(color_metric="rgb"):
            Applies cluster colors to nodes based on the provided cluster_color_dict.

        process_mp():
            Processes the graph by assigning data, cluster labels, colors and then removing unwanted data attributes.
    """

    def __init__(self, mp, df, attr_to_assign, cluster_label_dict=None, cluster_color_dict=None):
        self.mp = mp
        self.df = df
        self.attr_to_assign = attr_to_assign
        self.cluster_label_dict = cluster_label_dict
        self.cluster_color_dict = cluster_color_dict

    def _get_attribute_value(self, eid, attr, match_col):
        """Attempt to fetch an attribute value for a given eid; return 'nan' on failure."""
        try:
            return str(self.df[self.df[match_col] == eid][attr].values[0])
        except:
            return "nan"

    def _assign_attributes_for_family(self, node, eids, match_col):
        """Assign attributes to a node representing a family, concatenating values with '; '."""
        attr_dict = {att: [] for att in self.attr_to_assign}
        for eid in eids:
            for attr in self.attr_to_assign:
                attr_dict[attr].append(self._get_attribute_value(eid, attr, match_col))
        for k, v in attr_dict.items():
            self.mp.nodes[node][k] = ";".join(v)

    def _assign_single_node_attributes(self, node, match_id, match_col):
        """Assign attributes to a single node, handling non-family nodes."""
        for attr in self.attr_to_assign:
            self.mp.nodes[node][attr] = self._get_attribute_value(
                match_id, attr, match_col
            )

    def assign_data_to_mp(
        self,
        match_col="eid",
    ):
        """Assign data from df to mp nodes based on a matching column."""
        for node in self.mp.nodes:
            match_id = self.mp.nodes[node][match_col]
            if "family_" in self.mp.nodes[node]["label"]:
                eids = self.mp.nodes[node]["eid"].split(";")
                self._assign_attributes_for_family(node, eids, match_col)
            else:
                self._assign_single_node_attributes(node, match_id, match_col)
        # print("Data assigned to nodes.")
        # Remove multiple clusters if only 1 in families
        # find the attribute that contains the cluster information
        cluster_attribute = [
            attr
            for attr in list(self.mp.nodes(data=True))[0][1].keys()
            if attr.startswith("cluster")
        ][0]
        for node in self.mp.nodes(data=True):
            cluster = node[1][cluster_attribute]
            if ";" in cluster:
                clusters = set(cluster.split(";"))
                if len(clusters) == 1:
                    node[1][cluster_attribute] = list(clusters)[0]

        return self.mp

    def _remove_node_attribute(self, node, attr):
        """Remove a specific attribute from a node."""
        self.mp.nodes[node].pop(attr, None)

    def remove_data(self, attr_to_remove=["x", "y", "size", "shape"]):
        """Remove specified attributes from all nodes."""
        for node in self.mp.nodes():
            for attr in attr_to_remove:
                self._remove_node_attribute(node, attr)
        # print("Unwanted node attributes removed.")
        return self.mp
        
    def apply_cluster_labels(self):
        """
        Apply cluster labels to nodes based on the provided cluster_label_dict.
        This method identifies the cluster attribute in nodes and adds a corresponding
        'cluster_label' attribute based on the mapping provided in cluster_label_dict.
        """
        if not self.cluster_label_dict:
            return self.mp
            
        # Find the attribute that contains the cluster information
        cluster_keys = [
            attr
            for attr in list(self.mp.nodes(data=True))[0][1].keys()
            if attr.startswith("cluster")
        ]
        
        if not cluster_keys:
            return self.mp
            
        cluster_attribute = cluster_keys[0]
        
        # Apply cluster labels to each node
        for node in self.mp.nodes():
            cluster_id = self.mp.nodes[node][cluster_attribute]
            
            # Handle case where a node might have multiple clusters (separated by semicolons)
            if ";" in cluster_id:
                cluster_ids = cluster_id.split(";")
                cluster_labels = [self.cluster_label_dict.get(cid, "Unknown") for cid in cluster_ids]
                self.mp.nodes[node]["cluster_label"] = ";".join(cluster_labels)
            else:
                self.mp.nodes[node]["cluster_label"] = self.cluster_label_dict.get(cluster_id, "Unknown")
                
        return self.mp
    
    def apply_cluster_colors(self, color_metric="rgb"):
        """
        Apply cluster colors to nodes based on the provided cluster_color_dict.
        
        Args:
            color_metric (str): The color metric to use from the cluster_color_dict.
                                Options: "rgb", "hsv", or "alpha".
        """
        if not self.cluster_color_dict:
            return self.mp
            
        # Find the attribute that contains the cluster information
        cluster_keys = [
            attr
            for attr in list(self.mp.nodes(data=True))[0][1].keys()
            if attr.startswith("cluster")
        ]
        
        if not cluster_keys:
            return self.mp
            
        cluster_attribute = cluster_keys[0]
        
        # Apply cluster colors to each node
        for node in self.mp.nodes():
            cluster_id = self.mp.nodes[node][cluster_attribute]
            
            # For nodes with a single cluster
            if ";" not in cluster_id:
                if cluster_id in self.cluster_color_dict and color_metric in self.cluster_color_dict[cluster_id]:
                    self.mp.nodes[node]["color"] = self.cluster_color_dict[cluster_id][color_metric]
            # For nodes with multiple clusters (take color of first cluster)
            else:
                cluster_ids = cluster_id.split(";")
                first_cluster = cluster_ids[0]
                if first_cluster in self.cluster_color_dict and color_metric in self.cluster_color_dict[first_cluster]:
                    self.mp.nodes[node]["color"] = self.cluster_color_dict[first_cluster][color_metric]
                
        return self.mp

    def process_mp(self, match_col="eid", attr_to_remove=["x", "y", "size", "shape"], color_metric="rgb"):
        """
        Process the mp graph by assigning data, applying cluster labels and colors, 
        and then removing unwanted data.
        
        Args:
            match_col (str): Column to use for matching between DataFrame and graph.
            attr_to_remove (list): List of attributes to remove from nodes.
            color_metric (str): The color metric to use from the cluster_color_dict.
        """
        self.assign_data_to_mp(match_col)
        self.apply_cluster_labels()
        self.apply_cluster_colors(color_metric)
        self.remove_data(attr_to_remove)
        return self.mp


# example usage
# cluster_label_dict = {"0": "Topic A", "1": "Topic B", "2": "Topic C"}
# cluster_color_dict = {
#     "0": {
#         "rgb": [0.585, 0.270, 0.900],
#         "hsv": [0.75, 0.7, 0.900],
#         "alpha": 1.0
#     },
#     "1": {
#         "rgb": [0.2, 0.6, 0.8],
#         "hsv": [0.55, 0.75, 0.8],
#         "alpha": 1.0
#     }
# }
# data_assigner = MainPathDataAssigner(
#     graph, 
#     df, 
#     ["title", "cited_by", "doi", "year", "first_author"], 
#     cluster_label_dict, 
#     cluster_color_dict
# )
# graph = data_assigner.process_mp(color_metric="rgb")
# print(graph.nodes(data=True))