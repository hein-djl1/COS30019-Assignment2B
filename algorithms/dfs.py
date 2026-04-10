from node import SearchNode


def dfs(graph):
    start = graph.origin

    start_node = SearchNode(state=start, parent=None, action=None, path_cost=0)

    stack = [start_node]
    seen = {start.id}
    nodes_created = 1

    while stack:
        current = stack.pop()

        if graph.is_goal(current.state):
            return current.state.id, nodes_created, current.get_path()

        neighbours = graph.get_neighbours(current.state.id)

        for neighbour, cost in reversed(neighbours):
            if neighbour.id in seen:
                continue

            seen.add(neighbour.id)

            child = SearchNode(
                state=neighbour,
                parent=current,
                action=cost,
                path_cost=current.path_cost + cost
            )

            stack.append(child)
            nodes_created += 1

    return None, nodes_created, []
