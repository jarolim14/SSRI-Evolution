import networkx as nx


class MainPathReader:
    def __init__(self, filepath):
        self.filepath = filepath
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
        vertex_id = parts[0]  # ID of the vertex
        label = parts[1].strip('"')  # Label of the vertex, stripped of quotes
        x, y, size = map(float, parts[2:5])
        shape = parts[5]  # The shape of the vertex
        eid = parts[-1]  # The eid attribute
        self.graph.add_node(
            vertex_id, label=label, x=x, y=y, size=size, shape=shape, eid=eid
        )

    def parse_arc(self, line):
        parts = line.split()
        source, target = parts[0], parts[1]  # IDs of the source and target vertices
        weight = float(parts[2])  # The weight of the arc
        self.graph.add_edge(source, target, weight=weight)

    def get_graph(self):
        return self.graph

    # example usage
    # reader = MainPathReader("data/Graph1.net")
    # graph = reader.get_graph()
