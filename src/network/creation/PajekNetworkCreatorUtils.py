import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


class PajekNetworkCreatorUtils:
    @staticmethod
    def loop_to_df(node, df):
        loop_df_cols = [
            "unique_auth_year",
            "journal",
            "eid",
            "title",
            "citedby_count",
            "reference_eids",
            "nr_references",
            "filtered_reference_eids",
            "api_url",
            "doi",
        ]
        eids = node[1]["eid"].split(";")
        df_loop = df[df["eid"].isin(eids)][loop_df_cols]
        return df_loop

    # try finding a cycle
    @staticmethod
    def G_is_a_cycle(G):
        try:
            nx.find_cycle(G)
            return True
        except nx.exception.NetworkXNoCycle:
            return False

    @staticmethod
    def remove_reverse_time_bidirectional_edge(G):
        # Find and remove bidirectional edges where the earlier paper cites the later paper
        bidirectional_edges_removed = []
        bidirectional_edges = []
        for edge in G.edges():
            if edge[::-1] in G.edges():
                if edge not in bidirectional_edges:
                    bidirectional_edges.append(edge)

        for edge in bidirectional_edges:
            year1 = pd.to_numeric(edge[0].split("_")[1])
            year2 = pd.to_numeric(edge[1].split("_")[1])
            if year2 > year1:
                G.remove_edge(*edge)
                bidirectional_edges_removed.append(edge)
                print("Removed bidirectional edge: ", edge)

        return G, bidirectional_edges_removed

    @staticmethod
    def draw_graph(G):
        # remove self loops
        G.remove_edges_from(nx.selfloop_edges(G))
        # Define layout for the graph
        pos = nx.circular_layout(G)

        # Find cycles in the graph
        try:
            cycles = nx.find_cycle(G)
            cycle_edges = set(cycles)  # Convert to set for efficient lookup
        except nx.exception.NetworkXNoCycle:
            cycle_edges = set()

        # Plot the graph with cycle edges colored differently
        fig, ax = plt.subplots(figsize=(10, 10))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=50,
            node_color="skyblue",
            font_size=10,
            arrowsize=15,
            edge_color=[
                "red" if edge in cycle_edges else (0.53, 0.81, 0.92, 0.8)
                for edge in G.edges()
            ],  # Color edges red if part of a cycle, otherwise use default color
            ax=ax,
            alpha=0.9,  # Adjust node opacity; this doesn't affect edge opacity set by RGBA
        )
        plt.show()

    def create_graph_from_family(df_family):
        # Preprocess data
        label_eid_dict = dict(zip(df_family["eid"], df_family["unique_auth_year"]))

        # Initialize graph
        G = nx.DiGraph()

        # Add nodes to the graph
        nodes = df_family["unique_auth_year"].values
        G.add_nodes_from(nodes)

        # Construct edges for the graph
        edges = []
        for i, row in df_family.iterrows():
            ref_eids = row["filtered_reference_eids"]
            uay = row["unique_auth_year"]
            for ref_eid in ref_eids:
                if ref_eid in label_eid_dict:
                    ref_uay = label_eid_dict[ref_eid]
                    edges.append((uay, ref_uay))

        G.add_edges_from(edges)

        # draw the graph
        PajekNetworkCreatorUtils.draw_graph(G)

        # Print false time loop
        print("Reverse timed citation edges:")
        reversed_edges = []
        for edge in edges:
            year1 = pd.to_numeric(edge[0].split("_")[1])
            year2 = pd.to_numeric(edge[1].split("_")[1])
            if year2 > year1:
                print(edge)
                reversed_edges.append(edge)
        print("==================================")
        print(
            "Cochrane Reviews: ",
            "; ".join(
                [
                    str(row["unique_auth_year"])
                    for i, row in df_family.iterrows()
                    if "cochrane" in row["journal"].lower()
                ]
            ),
        )

        return G, reversed_edges
