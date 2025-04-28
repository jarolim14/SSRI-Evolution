import colorcet as cc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


class MainPathPlotterInteractive:
    """
    A class to create interactive network visualizations from NetworkX graphs with
    advanced clustering, labeling, and timeline features.
    
    Attributes:
        G (networkx.Graph): The graph to visualize.
        cluster_col (str): Column name containing cluster information.
        label_col (str): Column name containing label information.
        hover_cols (list): List of columns to display on hover.
        pos (dict): Dictionary of node positions.
    """
    
    def __init__(
        self, 
        G, 
        cluster_col="cluster_0", 
        label_col="label", 
        hover_cols=["title", "cited_by", "year", "first_author"],
        color_attr="color",
        cluster_label_attr="cluster_label"
    ):
        """
        Initialize the MainPathPlotterInteractive class.
        
        Args:
            G (networkx.Graph): The graph to visualize.
            cluster_col (str): Column name containing cluster information.
            label_col (str): Column name containing label information.
            hover_cols (list): List of columns to display on hover.
            color_attr (str): Attribute name for node colors (set by MainPathDataAssigner).
            cluster_label_attr (str): Attribute name for cluster labels.
        """
        self.G = G
        self.cluster_col = cluster_col
        self.label_col = label_col
        self.color_attr = color_attr
        self.cluster_label_attr = cluster_label_attr
        
        # Add cluster information to hover if they exist in the graph
        additional_cols = []
        if cluster_label_attr in next(iter(G.nodes(data=True)))[1]:
            additional_cols.append(cluster_label_attr)
        if "doi" in next(iter(G.nodes(data=True)))[1]:
            additional_cols.append("doi")
            
        self.hover_cols = hover_cols + additional_cols
        
        # Don't add these twice if they're already included
        if label_col not in self.hover_cols:
            self.hover_cols.append(label_col)
        if cluster_col not in self.hover_cols:
            self.hover_cols.append(cluster_col)
            
        self.pos = None

    @staticmethod
    def add_line_breaks(text, char_limit=100):
        """
        Add line breaks to long text for better hover display.
        
        Args:
            text (str): Text to format.
            char_limit (int): Maximum characters per line.
            
        Returns:
            str: Formatted text with line breaks.
        """
        if not isinstance(text, str):
            text = str(text)
            
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

    def adjust_overlap(self, min_dist=0.1, max_iterations=100):
        """
        Adjusts the positions to reduce overlap.

        Args:
            min_dist (float): Minimum desired distance between nodes.
            max_iterations (int): Maximum number of iterations to perform adjustments.
            
        Returns:
            dict: Adjusted positions.
        """
        for iteration in range(max_iterations):
            moved = False
            for node1 in self.pos:
                for node2 in self.pos:
                    if node1 != node2:
                        # Calculate the distance between two nodes
                        delta = np.array(self.pos[node1]) - np.array(self.pos[node2])
                        dist = np.sqrt(np.sum(delta**2))
                        # If the nodes are too close, push them apart
                        if dist < min_dist:
                            moved = True
                            displacement = delta / dist * (min_dist - dist) / 2
                            self.pos[node1] = self.pos[node1] + displacement
                            self.pos[node2] = self.pos[node2] - displacement
            # If no nodes were moved in this iteration, stop the adjustment
            if not moved:
                print(f"Adjustment finished after {iteration} iterations")
                break
                
        # Scale positions to fit well within the plot
        positions = np.array(list(self.pos.values()))
        x_min, y_min = positions.min(axis=0)
        x_max, y_max = positions.max(axis=0)
        
        # Normalize to fit within a reasonable range
        scale_factor = 1.0
        for node in self.pos:
            self.pos[node] = scale_factor * (self.pos[node] - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
            
        return self.pos

    def clean_years(self):
        """
        Convert year attribute to integer, handling multiple years by taking the mean.
        
        Returns:
            tuple: (min_year, max_year, nodes_sorted_by_year)
        """
        # Check if year attribute exists
        if "year" not in next(iter(self.G.nodes(data=True)))[1]:
            print("Warning: 'year' attribute not found in nodes")
            return None, None, list(self.G.nodes(data=True))
            
        # Convert year attribute to integer, if not already
        for node, data in self.G.nodes(data=True):
            if isinstance(data.get("year"), str):
                try:
                    # Handle multiple years separated by semicolons
                    years = data["year"].split(";")
                    valid_years = [int(y) for y in years if y.strip().isdigit()]
                    if valid_years:
                        data["year"] = int(sum(valid_years) / len(valid_years))
                    else:
                        data["year"] = 0  # Default for invalid years
                except ValueError:
                    data["year"] = 0  # Default for conversion errors
            elif data.get("year") is None:
                data["year"] = 0

        # Sort nodes based on year attribute
        nodes_sorted_by_year = sorted(
            self.G.nodes(data=True), key=lambda x: x[1].get("year", 0), reverse=True
        )

        # Get the range of years
        years = [data.get("year", 0) for _, data in nodes_sorted_by_year]
        min_year = min(years) if years else 0
        max_year = max(years) if years else 0

        return min_year, max_year, nodes_sorted_by_year

    def hover_texts(self):
        """
        Generate formatted hover texts for each node.
        
        Returns:
            list: List of HTML-formatted hover texts.
        """
        hover_texts = []
        for _, data in self.G.nodes(data=True):
            col_text_dict = {}
            for col in self.hover_cols:
                if col in data and data[col] is not None:
                    col_text_dict[col] = self.add_line_breaks(data[col])
            
            # Add DOI as a clickable link if present
            if "doi" in col_text_dict:
                doi = col_text_dict["doi"]
                if doi != "nan" and doi.strip():
                    col_text_dict["doi"] = f'<a href="https://doi.org/{doi}" target="_blank">{doi}</a>'
            
            # Format the hover text
            node_hover_text = "<br>".join(
                [f"<b>{col}:</b> {col_text_dict[col]}" for col in col_text_dict]
            )
            hover_texts.append(node_hover_text)
        return hover_texts

    def get_node_colors(self):
        """
        Get node colors from the color attribute if available, otherwise generate colors based on clusters.
        
        Returns:
            list: List of colors for each node.
        """
        # Check if color attribute exists from MainPathDataAssigner
        if self.color_attr in next(iter(self.G.nodes(data=True)))[1]:
            # Use the RGB values assigned by MainPathDataAssigner
            colors = []
            for _, data in self.G.nodes(data=True):
                rgb_values = data.get(self.color_attr)
                if isinstance(rgb_values, list) and len(rgb_values) == 3:
                    # Convert RGB values (0-1 scale) to hex color
                    r, g, b = rgb_values
                    color_hex = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                    colors.append(color_hex)
                else:
                    # Default color if RGB values are not available
                    colors.append("#808080")  # Gray
            return colors
        else:
            # Generate colors based on clusters
            return self._generate_cluster_colors()
    
    def _generate_cluster_colors(self):
        """
        Generate colors based on cluster values.
        
        Returns:
            list: List of colors for each node.
        """
        clusters = [data.get(self.cluster_col, "0") for _, data in self.G.nodes(data=True)]
        
        # Handle multi-cluster nodes (take the first cluster)
        processed_clusters = []
        for cluster in clusters:
            if isinstance(cluster, str) and ";" in cluster:
                processed_clusters.append(cluster.split(";")[0])
            else:
                processed_clusters.append(str(cluster))
        
        unique_clusters = sorted(set(processed_clusters))
        
        # Using colorcet to get a colormap with enough distinct colors
        color_list = cc.glasbey_dark[: len(unique_clusters)]
        
        # Creating a color dictionary mapping each cluster to a color
        color_dict = {
            cluster: color_list[i] for i, cluster in enumerate(unique_clusters)
        }
        
        # Generate colors for each node
        return [color_dict[cluster] for cluster in processed_clusters]

    def node_labels(self):
        """
        Get node labels.
        
        Returns:
            list: List of node labels.
        """
        return [data.get(self.label_col, "") for _, data in self.G.nodes(data=True)]

    def edge_positions(self):
        """
        Get edge positions for plotting.
        
        Returns:
            tuple: (edge_x, edge_y) coordinates.
        """
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
        """
        Create the edge trace for plotting.
        
        Returns:
            go.Scatter: Plotly scatter trace for edges.
        """
        edge_x, edge_y = self.edge_positions()
        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.7, color="#888888"),
            hoverinfo="none",
            mode="lines",
        )
        return edge_trace

    def get_node_trace(self):
        """
        Create the node trace for plotting.
        
        Returns:
            go.Scatter: Plotly scatter trace for nodes.
        """
        node_x, node_y = zip(*self.pos.values())
        hover_texts = self.hover_texts()
        colors = self.get_node_colors()
        node_labels = self.node_labels()
        
        # Calculate node sizes based on node degree (or another metric)
        degrees = dict(self.G.degree())
        node_sizes = [10 + (degrees[node] * 3) for node in self.G.nodes()]
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=hover_texts,
            text=node_labels,
            textposition="middle center",
            textfont=dict(
                size=8,
                color='black'
            ),
            marker=dict(
                showscale=False,
                opacity=0.85,
                size=node_sizes,
                color=colors,
                line=dict(
                    width=1,
                    color='rgba(0,0,0,0.4)'
                )
            ),
        )
        return node_trace

    def create_legend_trace(self):
        """
        Create legend traces based on cluster labels.
        
        Returns:
            list: List of Plotly scatter traces for legend.
        """
        # Check if cluster label attribute exists
        if self.cluster_label_attr not in next(iter(self.G.nodes(data=True)))[1]:
            return []
            
        # Get cluster labels and colors
        cluster_data = {}
        for _, data in self.G.nodes(data=True):
            cluster_id = data.get(self.cluster_col, "0")
            if isinstance(cluster_id, str) and ";" in cluster_id:
                cluster_id = cluster_id.split(";")[0]
                
            cluster_label = data.get(self.cluster_label_attr, "Unknown")
            if isinstance(cluster_label, str) and ";" in cluster_label:
                cluster_label = cluster_label.split(";")[0]
                
            # Get color from the node
            if self.color_attr in data:
                rgb_values = data.get(self.color_attr)
                if isinstance(rgb_values, list) and len(rgb_values) == 3:
                    r, g, b = rgb_values
                    color = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                    cluster_data[cluster_id] = (cluster_label, color)
        
        # Create legend traces
        legend_traces = []
        for cluster_id, (label, color) in cluster_data.items():
            legend_trace = go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f"{label} (Cluster {cluster_id})",
                showlegend=True
            )
            legend_traces.append(legend_trace)
            
        return legend_traces

    def plot_network_on_timeline_interactive(
        self, 
        savingpath=None, 
        return_fig=True, 
        adjust_overlap=True, 
        pos=None,
        show_legend=True,
        title=None
    ):
        """
        Create and display an interactive network visualization.
        
        Args:
            savingpath (str, optional): Path to save the visualization.
            return_fig (bool): Whether to return the figure as HTML.
            adjust_overlap (bool): Whether to adjust node positions to reduce overlap.
            pos (dict, optional): Pre-calculated node positions.
            show_legend (bool): Whether to show the cluster legend.
            title (str, optional): Plot title.
            
        Returns:
            str or None: HTML representation of figure if return_fig is True.
        """
        # Set node positions
        if not pos:
            self.pos = nx.kamada_kawai_layout(
                self.G,
                weight=None,
                scale=5.0,
                center=None,
                dim=2,
            )
        else:
            self.pos = pos

        # Adjust node positions to reduce overlap if requested
        if adjust_overlap:
            self.pos = self.adjust_overlap(0.15, 100)

        # Get edge and node traces
        edge_trace = self.get_edge_trace()
        node_trace = self.get_node_trace()
        
        # Create legend traces if requested
        legend_traces = self.create_legend_trace() if show_legend else []
        
        # Set dimensions based on graph size
        graph_size = len(self.G.nodes)
        width = 1000
        height = 800
        
        if graph_size > 100:
            width = 1200
            height = 900
        if graph_size > 200:
            width = 1500
            height = 1000
        
        # Create the figure
        fig = go.Figure(
            data=[edge_trace, node_trace] + legend_traces,
            layout=go.Layout(
                title=title,
                #titlefont=dict(size=16),
                width=width,
                height=height,
                showlegend=show_legend,
                hovermode="closest",
                hoverlabel=dict(
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="rgba(0, 0, 0, 0.4)",
                    font=dict(color="black", size=12),
                ),
                margin=dict(b=40, l=40, r=40, t=60),
                xaxis=dict(
                    showline=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    visible=False,
                ),
                yaxis=dict(
                    showline=False,
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                    visible=False,
                ),
                legend=dict(
                    x=0.01,
                    y=0.99,
                    bgcolor="rgba(255, 255, 255, 0.8)",
                    bordercolor="rgba(0, 0, 0, 0.2)",
                    borderwidth=1
                ),
                dragmode="zoom",
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
            ),
        )

        # Save the figure if savingpath is provided
        if savingpath:
            if savingpath.endswith(".html"):
                fig.write_html(savingpath)
            elif savingpath.endswith(".png"):
                fig.write_image(savingpath, scale=2, width=width, height=height)
            else:
                fig.write_html(f"{savingpath}.html")
                
        # Show the figure
        fig.show()
        
        # Return HTML representation if requested
        if return_fig:
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")
    
    def plot_timeline_view(self, savingpath=None, return_fig=True):
        """
        Create a timeline view of the network, organizing nodes by year.
        
        Args:
            savingpath (str, optional): Path to save the visualization.
            return_fig (bool): Whether to return the figure as HTML.
            
        Returns:
            str or None: HTML representation of figure if return_fig is True.
        """
        # Clean years and get year range
        min_year, max_year, nodes_sorted_by_year = self.clean_years()
        
        if min_year is None or max_year is None:
            print("Warning: Year information not available for timeline view")
            return None
        
        # Prepare node data
        node_x = []
        node_y = []
        node_colors = []
        node_sizes = []
        node_labels = []
        hover_texts = []
        
        # Get colors using existing method
        color_dict = {}
        for i, (node, data) in enumerate(self.G.nodes(data=True)):
            # Get cluster and color
            cluster_id = data.get(self.cluster_col, "0")
            if isinstance(cluster_id, str) and ";" in cluster_id:
                cluster_id = cluster_id.split(";")[0]
                
            # Use assigned color if available
            if self.color_attr in data:
                rgb_values = data.get(self.color_attr)
                if isinstance(rgb_values, list) and len(rgb_values) == 3:
                    r, g, b = rgb_values
                    color = f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
                    color_dict[node] = color
        
        # If no colors assigned, generate them
        if not color_dict:
            base_colors = self._generate_cluster_colors()
            for i, node in enumerate(self.G.nodes()):
                color_dict[node] = base_colors[i]
        
        # Prepare hover texts
        hover_dict = {}
        for i, (node, data) in enumerate(self.G.nodes(data=True)):
            col_text_dict = {}
            for col in self.hover_cols:
                if col in data and data[col] is not None:
                    col_text_dict[col] = self.add_line_breaks(data[col])
            
            node_hover_text = "<br>".join(
                [f"<b>{col}:</b> {col_text_dict[col]}" for col in col_text_dict]
            )
            hover_dict[node] = node_hover_text
        
        # Organize nodes by year
        year_nodes = {}
        for node, data in self.G.nodes(data=True):
            year = data.get("year", 0)
            if year not in year_nodes:
                year_nodes[year] = []
            year_nodes[year].append(node)
        
        # Position nodes on timeline
        year_range = max_year - min_year
        if year_range == 0:
            year_range = 1  # Avoid division by zero
            
        for year, nodes in year_nodes.items():
            # Normalize year to x position
            x_pos = (year - min_year) / year_range
            
            # Distribute nodes vertically
            for i, node in enumerate(nodes):
                node_x.append(x_pos)
                node_y.append(0.1 + 0.8 * (i / max(1, len(nodes) - 1)))
                node_colors.append(color_dict[node])
                node_sizes.append(10 + (self.G.degree(node) * 3))
                node_labels.append(self.G.nodes[node].get(self.label_col, ""))
                hover_texts.append(hover_dict[node])
        
        # Create node trace
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode="markers+text",
            hoverinfo="text",
            hovertext=hover_texts,
            text=node_labels,
            textposition="top center",
            textfont=dict(size=8),
            marker=dict(
                showscale=False,
                opacity=0.85,
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='rgba(0,0,0,0.4)')
            ),
        )
        
        # Create year markers for x-axis
        year_markers = list(range(min_year, max_year + 1, max(1, (max_year - min_year) // 10)))
        
        # Create the figure
        fig = go.Figure(
            data=[node_trace],
            layout=go.Layout(
                title="Publication Timeline",
                width=1200,
                height=600,
                showlegend=False,
                hovermode="closest",
                hoverlabel=dict(
                    bgcolor="rgba(255, 255, 255, 0.95)",
                    bordercolor="rgba(0, 0, 0, 0.4)",
                    font=dict(color="black")
                ),
                xaxis=dict(
                    title="Year",
                    tickvals=[(year - min_year) / year_range for year in year_markers],
                    ticktext=[str(year) for year in year_markers],
                    showgrid=True,
                    zeroline=True,
                    gridcolor='rgba(220, 220, 220, 0.8)',
                ),
                yaxis=dict(
                    showgrid=False,
                    zeroline=False,
                    showticklabels=False,
                ),
                dragmode="zoom",
                paper_bgcolor='rgba(255,255,255,1)',
                plot_bgcolor='rgba(255,255,255,1)',
            ),
        )
        
        # Save the figure if savingpath is provided
        if savingpath:
            if savingpath.endswith(".html"):
                fig.write_html(savingpath)
            elif savingpath.endswith(".png"):
                fig.write_image(savingpath, scale=2, width=1200, height=600)
            else:
                fig.write_html(f"{savingpath}_timeline.html")
                
        # Show the figure
        fig.show()
        
        # Return HTML representation if requested
        if return_fig:
            return pio.to_html(fig, full_html=False, include_plotlyjs="cdn")


# example usage
# cluster_label_dict = {"0": "Topic A", "1": "Topic B", "2": "Topic C"}
# cluster_color_dict = {"0": {"rgb": [0.585, 0.270, 0.900]}, "1": {"rgb": [0.2, 0.6, 0.8]}}
# 
# # First assign data, labels and colors
# data_assigner = MainPathDataAssigner(
#     graph, 
#     df, 
#     ["title", "cited_by", "doi", "year", "first_author"], 
#     cluster_label_dict, 
#     cluster_color_dict
# )
# graph = data_assigner.process_mp()
# 
# # Then create the interactive plot
# plotter = MainPathPlotterInteractive(
#     graph, 
#     cluster_col="cluster_0", 
#     label_col="label",
#     hover_cols=["title", "cited_by", "year", "first_author"],
#     color_attr="color",
#     cluster_label_attr="cluster_label"
# )
# 
# # Show network visualization
# plotter.plot_network_on_timeline_interactive(
#     return_fig=False, 
#     adjust_overlap=True,
#     show_legend=True,
#     title="Research Topic Network"
# )
# 
# # Show timeline view
# plotter.plot_timeline_view()
# 
# # Save visualizations
# plotter.plot_network_on_timeline_interactive(savingpath="network_visualization.html")
# plotter.plot_timeline_view(savingpath="timeline_visualization.html")