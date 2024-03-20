import networkx as nx


class MainPathReader:
    def __init__(self, filepath, node_attributes=None):
        self.filepath = filepath
        self.node_attributes = node_attributes
        self.graph = nx.DiGraph()
        self.read_and_parse_file()
        print(self.graph)

    def read_and_parse_file(self):
        mode = None  # Track the current section being parsed (vertices or arcs)
        with open(self.filepath, "r") as file:
            for line in file:
                line = line.strip()
                if line.startswith("*Vertices"):  # Start of the vertices section
                    mode = "vertices"
                    continue
                elif line.startswith("*Arcs"):  # Start of the arcs section
                    mode = "arcs"
                    continue

                if mode == "vertices" and line:
                    self.parse_vertex(line)
                elif mode == "arcs" and line:
                    self.parse_arc(line)

    def parse_vertex(self, line):
        parts = line.split()
        # drop 'eid' and 'unique_auth_year' from parts
        parts = [p for p in parts if p not in self.node_attributes]
        vertex_data = [
            "id",
            "label",
            "x",
            "y",
            "size",
            "shape",
        ]
        vertex_data = vertex_data + self.node_attributes
        # create a dictionary of the vertex data
        vertex_dict = dict(zip(vertex_data, parts))
        # add the vertex to the graph
        self.graph.add_node(
            vertex_dict["id"],
            label=vertex_dict["label"],
            x=vertex_dict["x"],
            y=vertex_dict["y"],
            size=vertex_dict["size"],
            shape=vertex_dict["shape"],
            eid=vertex_dict["eid"],
            unique_auth_year=vertex_dict["unique_auth_year"],
        )

    def parse_arc(self, line):
        parts = line.split()
        source, target = parts[0], parts[1]  # IDs of the source and target vertices
        weight = float(parts[2])  # The weight of the arc
        self.graph.add_edge(source, target, weight=weight)

    def get_graph(self):
        return self.graph


# example usage
