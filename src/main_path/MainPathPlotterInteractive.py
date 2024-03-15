import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objects as go


class MainPathPlotterInteractive:
    def __init__(self, G, cluster_col):
        self.G = G
        self.cluster_col = cluster_col

    @staticmethod
    def add_line_breaks(text, char_limit=100):
        words = text.split()
        line = ""
        lines = []
        for word in words:
            if len(line + " " + word) <= char_limit:
                line += " " + word
            else:
                lines.append(line.strip())
                line = word
        if line:
            lines.append(line.strip())
        return "<br>".join(lines)

    def clean_years(self):
        """
        Convert year attribute to integer, if not already
        if multiple years are present, the mean is taken
        """
        # Convert year attribute to integer, if not already
        for node, data in self.G.nodes(data=True):
            if isinstance(data["year"], str):
                mean_year = sum(map(int, data["year"].split(";"))) / len(
                    data["year"].split(";")
                )
                data["year"] = int(mean_year)

        # Sort nodes based on year attribute
        nodes_sorted_by_year = sorted(
            self.G.nodes(data=True), key=lambda x: x[1]["year"], reverse=True
        )

        # Get the range of years
        min_year = min([data["year"] for node, data in nodes_sorted_by_year])
        max_year = max([data["year"] for node, data in nodes_sorted_by_year])

        return min_year, max_year, nodes_sorted_by_year

    def hover_texts(self):
        hover_texts = []
        for node, data in self.G.nodes(data=True):
            wrapped_title = self.add_line_breaks(data["title"])
            wrapped_title = f"Cluster: {data[self.cluster_col]}<br>{wrapped_title}"
            hover_texts.append(wrapped_title)
        return hover_texts

    def cluster_color_dict(self):
        """
        Creates a dictionary mapping clusters to distinct colors using the colorcet library,
        and returns a list of colors for each node based on its cluster.

        Returns:
            list: A list of color strings in hex format for each node in the graph, based on its cluster.
        """
        clusters = nx.get_node_attributes(self.G, self.cluster_col).values()
        unique_clusters = sorted(set(clusters))

        # Using colorcet to get a colormap with enough distinct colors
        # Adjust the colormap if needed to suit your preferences for color variety and distinctness
        color_list = cc.glasbey_dark[: len(unique_clusters)]

        # Creating a color dictionary mapping each cluster to a color
        color_dict = {
            cluster: color_list[i] for i, cluster in enumerate(unique_clusters)
        }

        # Generating a list of colors for each node based on its cluster
        colors = [color_dict[cluster] for cluster in clusters]
        return colors

    def node_labels(self, feature="year"):
        node_labels = []
        for node, data in self.G.nodes(data=True):
            node_labels.append(data[feature])
        return node_labels

    def edge_positions(self):
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = self.pos[edge[0]]
            x1, y1 = self.pos[edge[1]]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)
        return edge_x, edge_y

    def get_edge_trace(self):
        edge_x, edge_y = self.edge_positions()
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
        )
        return edge_trace

    def get_node_trace(self):
        node_x, node_y = zip(*self.pos.values())
        hover_texts = self.hover_texts()
        colors = self.cluster_color_dict()
        node_labels = self.node_labels()
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=hover_texts,
            text=node_labels,  # Permanent labels for nodes
            textposition="middle center",
            marker=dict(
                showscale=False,
                opacity=0.3,  # Decreased opacity of nodes for the desired transparency
                # colorscale='YlGnBu',
                size=20,
                colorbar=dict(
                    thickness=15,
                    title="Node Connections",
                    xanchor="left",
                    titleside="right",
                ),
                line_width=2,
                color=colors,
            ),
        )
        return node_trace

    def plot_network_on_timeline_interactive(self, savingpath=None, return_fig=False):
        # Convert year attribute to integer, if not already
        # min_year, max_year, nodes_sorted_by_year = self.clean_years()

        self.pos = nx.kamada_kawai_layout(
            self.G, weight=None, scale=5.0, center=None, dim=2
        )  # Use Kamada-Kawai layout for better node positioning

        edge_trace = self.get_edge_trace()

        node_trace = self.get_node_trace()
        width = 900
        height = 1500
        if len(self.G.nodes) > 190:
            width = 1300
            height = 1500
        # Figure Definition
        fig = go.Figure(
            data=[edge_trace, node_trace],
            layout=go.Layout(
                width=width,  # Example width
                height=height,  # Example height
                showlegend=False,  # Adjust based on your need to show or hide the legend
                hovermode="closest",
                hoverlabel=dict(  # Customizing the hover label's appearance
                    bgcolor="rgba(249, 98, 186, 0.11)",  # Setting the background color to a semi-transparent gray
                    bordercolor="rgba(0, 0, 0, 0.14)",
                    font=dict(color="black"),  # Text color in the hover label
                ),
                margin=dict(b=0, l=0, r=0, t=40),
                xaxis=dict(
                    showline=False,  # Hides the axis line
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,  # Hide tick labels
                    visible=False,  # Makes the whole axis (including the space for it) invisible
                ),
                yaxis=dict(
                    showline=False,  # Hides the axis line
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,  # Hide tick labels
                    visible=False,  # Makes the whole axis (including the space for it) invisible
                ),
                dragmode="zoom",
            ),
        )

        plt.tight_layout()
        if savingpath:
            if savingpath.endswith(".html"):  # Use 'savingpath' consistently
                fig.write_html(savingpath)
            elif savingpath.endswith(".png"):
                fig.write_image(savingpath, dpi=300)
        fig.show()
        if return_fig:
            return fig


# example usage
# plotter = MainPathPlotterInteractive(graph, "cluster")
# plotter.plot_network_on_timeline_interactive()
# plotter.plot_network_on_timeline_interactive(savingpath="network_timeline.html")
# plotter.plot_network_on_timeline_interactive(savingpath="network_timeline.png")
