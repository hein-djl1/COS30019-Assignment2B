from graph import Graph


def parse_file(filepath):
    """Read the file and return a Graph object."""

    with open(filepath, "r") as f:
        lines = f.readlines()

    graph = Graph()

    #figure out which section we are currently in
    current_section = None
    #go through all the lines in the file
    for line in lines:
        line = line.strip()
        if not line:
            continue

        #check for section headers
        if line == "Nodes:":
            current_section = "nodes"
        elif line == "Edges:":
            current_section = "edges"
        elif line == "Origin:":
            current_section = "origin"
        elif line == "Destinations:":
            current_section = "destinations"

        elif current_section == "nodes":
            node_id, coords = line.split(":")
            node_id = int(node_id.strip())
            coords = coords.strip().strip("()")
            x, y = coords.split(",")
            graph.add_node(node_id, float(x), float(y))

        elif current_section == "edges":
            from_to, cost = line.split(":")
            # from_to = "(2,1)"   cost = " 4"
            node_pair = from_to.strip().strip("()")
            from_id, to_id = node_pair.split(",")
            cost = float(cost.strip())
            graph.add_edge(int(from_id), int(to_id), cost)

        elif current_section == "origin":
            origin_id = int(line)
            graph.origin = graph.nodes[origin_id]

        elif current_section == "destinations":
            dest_ids = line.split(";")
            for dest_id in dest_ids:
                dest_id = dest_id.strip()
                if dest_id:
                    graph.destinations.append(graph.nodes[int(dest_id)])

    return graph
