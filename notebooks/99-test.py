import colorcet as cc
import networkx as nx
import pandas as pd
import sys
from dash import Dash, html
import colorcet as cc

sys.path.append("/Users/jlq293/Projects/Study-1-Bibliometrics/src/main_path/")


params = "alpha0.3_k20_res0.005"
p = f"../data/06-clustered-df/{params}.pkl"
df = pd.read_pickle(p)

path_to_main_path = "../data/08-main-paths/final_mp_clustered.graphml"
G = nx.read_graphml(path_to_main_path)

list(G.nodes(data=True))[0]


def node_and_edge_list_fun(G):
    nodes_list = []
    for node, data in G.nodes(data=True):
        dat = {
            "id": node,
            "cluster": data["cluster_alpha0.3_k20_res0.005"],
            "cluster_label": data["full_label"],
            "title": data["title"],
        }
        pos = {"x": 0, "y": 0}
        nodes_list.append({"data": dat, "position": pos})

    edges_list = []

    for source, target in G.edges:
        edges_list.append({"data": {"source": target, "target": source}})
    print("Node example")
    print(nodes_list[0])
    return nodes_list, edges_list


def default_stylesheet_fun(cluster_color_dict):
    default_stylesheet = [
        {
            "selector": "node",  # Select all nodes
            "style": {
                "shape": "rectangle",  # Change the shape to rectangle
                "width": "80px",  # Specify width of the nodes
                "height": "30px",  # Specify height of the nodes
                "label": "data(cluster)",  # Show node label
                "font-size": "24px",  # Set font size
                "text-valign": "center",
                "text-halign": "center",
            },
        },
        {
            "selector": "edge",
            "style": {
                # The default curve style does not work with certain arrows
                "curve-style": "bezier",
                "source-arrow-shape": "triangle",
                # "target-arrow-shape": "triangle",
            },
        },
    ]

    # Add styles for each label-color mapping
    for cluster, color in cluster_color_dict.items():
        default_stylesheet.append(
            {
                "selector": f'[cluster = "{cluster}"]',  # Select nodes with the specific label
                "style": {"background-color": color, "line-color": color},
            }
        )
    return default_stylesheet


def cluster_color_dict(G):
    cluster_color_dict = {
        cluster: cc.glasbey_light[i]
        for i, cluster in enumerate(
            set(nx.get_node_attributes(G, "cluster_alpha0.3_k20_res0.005").values())
        )
    }
    # print first
    print(f"Cluster color dictionary example: {list(cluster_color_dict.items())[:5]}")

    return cluster_color_dict


nodes_list, edges_list = node_and_edge_list_fun(G)

cluster_color_dict = cluster_color_dict(G)

default_stylesheet = default_stylesheet_fun(cluster_color_dict)


from dash import Dash, html
import dash_cytoscape as cyto

# nodes_list[0] = {'data': {'id': '1', 'label': '36', 'cluster': '36'}, 'position': {'x': 0, 'y': 0}}
cyto.load_extra_layouts()

app = Dash(__name__)
server = app.server


elements = nodes_list + edges_list

app.layout = html.Div(
    [
        html.Div(
            cyto.Cytoscape(
                id="main-path-cytoscape",
                elements=elements,
                style={
                    "width": "100%",
                    "height": "800px",
                    "background-color": "#f5f5f5",  # Change background color here
                    "padding": "20px",
                },
                layout={"name": "dagre"},
                stylesheet=default_stylesheet,
                #                responsive=True,
            ),
        ),
        html.Button("Print elements JSONified", id="button-main-path-cytoscape"),
        html.Div(id="html-main-path-cytoscape"),
    ]
)


@app.callback(
    Output("html-main-path-cytoscape", "children"),
    [Input("button-main-path-cytoscape", "n_clicks")],
    [State("main-path-cytoscape", "elements")],
)
def testCytoscape(n_clicks, elements):
    if n_clicks:
        return json.dumps(elements)


if __name__ == "__main__":
    app.run(debug=True)
