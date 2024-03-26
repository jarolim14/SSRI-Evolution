import networkx as nx


class PajekNetworkCreator:
    def __init__(self, G, attr_to_keep):
        self.OG = G
        self.NG = G.copy()
        self.log = {}
        self.attr_to_keep = attr_to_keep

    def prepare_attributes(self):
        for node, data in self.NG.nodes(data=True):
            keys_to_remove = [k for k in data if k not in self.attr_to_keep]
            for k in keys_to_remove:
                data.pop(k, None)
        for node, data in self.NG.nodes(data=True):
            for k, v in data.items():
                data[k] = str(v)

    def remove_loops(self):
        sccs = list(nx.strongly_connected_components(self.NG))
        Gloopless = nx.DiGraph()
        self.original_to_family = {}
        removed_eids = []  # To store eids of nodes merged into families

        for scc in sccs:
            if len(scc) > 1:
                eids = ";".join(
                    [
                        str(self.NG.nodes[node]["eid"])
                        for node in scc
                        if "eid" in self.NG.nodes[node]
                    ]
                )
                unique_auth_year = ";".join(
                    [
                        str(self.NG.nodes[node]["unique_auth_year"])
                        for node in scc
                        if "unique_auth_year" in self.NG.nodes[node]
                    ]
                )
                family_node = "family_" + "_".join(sorted([str(node) for node in scc]))
                Gloopless.add_node(
                    family_node, eid=eids, unique_auth_year=unique_auth_year
                )
                removed_eids.append(eids)  # Log merged eids
                for node in scc:
                    self.original_to_family[node] = family_node
            else:
                node = next(iter(scc))
                Gloopless.add_node(node, **self.NG.nodes[node])
                self.original_to_family[node] = node

        for u, v, data in self.NG.edges(data=True):
            new_u = self.original_to_family.get(u, u)
            new_v = self.original_to_family.get(v, v)
            if new_u != new_v:
                Gloopless.add_edge(new_u, new_v, **data)

        removed_eids_list = [
            eid for sublist in removed_eids for eid in sublist.split(";")
        ]
        self.NG = Gloopless
        self.log["loops_removed"] = {
            "count": len(removed_eids),
            "eids": removed_eids_list,
        }

    def remove_isolates(self):
        isolates = list(nx.isolates(self.NG))
        isolated_eids = [self.NG.nodes[iso]["eid"] for iso in isolates]
        self.NG.remove_nodes_from(isolates)
        isolated_eids_list = [
            eid for sublist in isolated_eids for eid in sublist.split(";")
        ]
        self.log["isolates_removed"] = {
            "count": len(isolates),
            "eids": isolated_eids_list,
        }

    def extract_largest_wcc(self):
        largest_wcc = max(nx.weakly_connected_components(self.NG), key=len)
        removed_nodes = set(self.NG.nodes()) - set(largest_wcc)
        removed_eids = [self.NG.nodes[node]["eid"] for node in removed_nodes]
        self.NG = self.NG.subgraph(largest_wcc).copy()
        removed_eids_list = [
            eid for sublist in removed_eids for eid in sublist.split(";")
        ]

        self.log["largest_wcc_removed"] = {
            "count": len(removed_nodes),
            "eids": removed_eids_list,
        }

    def prepare_pajek(self):
        self.prepare_attributes()
        self.remove_loops()
        self.remove_isolates()
        self.extract_largest_wcc()
        return self.NG, self.log


# example usage
# G = nx.read_gpickle("data/processed/merged_network.gpickle")
# attr_to_keep = ["eid", "unique_auth_year"]
# creator = PajekNetworkCreator(G, attr_to_keep)
# NG, log = creator.prepare_pajek()
# nx.write_pajek(NG, "data/processed/merged_network.net")
