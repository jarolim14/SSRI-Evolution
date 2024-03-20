class MainPathDataAssigner:
    """
    A class to assign data from a pandas DataFrame to nodes of a networkx graph based on a matching column,
    and to modify node attributes within the graph.

    Attributes:
        mp (networkx.Graph): The graph to which data will be added or modified.
        df (pandas.DataFrame): The DataFrame from which data will be sourced.

    Methods:
        assign_data_to_mp(attr_to_assign=["title", "cited_by", "doi", "year", "first_author"], match_col="eid"):
            Assigns attributes from the DataFrame to corresponding nodes in the graph.

        remove_data(attr_to_remove=["x", "y", "size", "shape"]):
            Removes specified attributes from all nodes in the graph.

        process_mp():
            Processes the graph by assigning data from the DataFrame and then removing unwanted data attributes.
    """

    def __init__(self, mp, df, attr_to_assign):
        self.mp = mp
        self.df = df
        self.attr_to_assign = attr_to_assign

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

    def process_mp(self):
        """Process the mp graph by assigning data and then removing unwanted data."""
        self.assign_data_to_mp()
        self.remove_data()
        return self.mp


# example usage
# data_assigner = MainPathDataAssigner(graph, df, ["title", "cited_by", "doi", "year", "first_author"])
# graph = data_assigner.process_mp()
# print(graph.nodes(data=True))
