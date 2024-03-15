import matplotlib.pyplot as plt
import networkx as nx


class MainPathPlotterStatic:
    def __init__(self, G):
        self.G = G

    def clean_years(self):
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

    def plot_network_on_timeline_vertical_static(
        self, savingpath=None, vertical_spacing=10.0, horizontal_spacing=1.0
    ):
        print("Plotting network on a static vertical timeline...")

        min_year, max_year, nodes_sorted_by_year = self.clean_years()

        plt.figure(figsize=(12, 20))
        pos = {}

        max_nodes_per_year = max(
            [
                sum(1 for _, data in nodes_sorted_by_year if data["year"] == year)
                for year in range(min_year, max_year + 1)
            ]
        )

        # Create a position for each node, adjusting for spacing settings
        current_vertical_position = 0
        for year in range(max_year, min_year - 1, -1):
            nodes_in_year = [
                node for node, data in nodes_sorted_by_year if data["year"] == year
            ]

            # Calculate the horizontal offset to center the nodes for the given year
            offset = (max_nodes_per_year - len(nodes_in_year)) * horizontal_spacing / 2

            for j, node in enumerate(nodes_in_year):
                pos[node] = (
                    j * horizontal_spacing + offset,
                    current_vertical_position * vertical_spacing,
                )
            current_vertical_position += 1

        # Draw nodes and edges
        nx.draw_networkx_nodes(
            self.G, pos, node_color="skyblue", alpha=0.5, node_size=400
        )
        nx.draw_networkx_edges(
            self.G, pos, alpha=1, arrowsize=15, arrowstyle="->", width=0.50
        )

        # Draw labels
        for node, (x, y) in pos.items():
            plt.text(
                x,
                y,
                self.G.nodes[node]["year"],
                size=12,
                ha="center",
                va="center",
            )
            # plt.text(x, y-2, self.G.nodes[node]['title'], size=6, ha="center", va="center")

        # Set ticks and labels on the y-axis for years
        plt.yticks(
            ticks=[i * vertical_spacing for i in range(max_year - min_year + 1)],
            labels=list(range(max_year, min_year - 1, -1)),
        )
        plt.xticks(
            ticks=[i * horizontal_spacing for i in range(max_nodes_per_year)],
            labels=list(range(1, max_nodes_per_year + 1)),
        )
        # plt.ylabel('Year')
        # plt.xlabel('Node Index')
        # plt.title('Network Timeline')

        # Remove the unwanted spines
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)

        plt.tight_layout()
        if savingpath:
            plt.savefig(savingpath, dpi=300, bbox_inches="tight")
        plt.show()


# example usage
# G = nx.read_gpickle("data/processed/processed_network.gpickle")
# mpp = MainPathPlotterStatic(G)
# mpp.plot_network_on_timeline_vertical_static(savingpath="data/processed/static_timeline.png")
# mpp.plot_network_on_timeline_vertical_static(savingpath="data/processed/static_timeline.png", vertical_spacing=20, horizontal_spacing=2)
