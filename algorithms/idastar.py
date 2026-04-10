from node import SearchNode
from heuristic import nearest_goal_dist


def search(node, graph, goals, threshold, path_ids, nodes_created):
    h = nearest_goal_dist(node.state, goals)
    f = node.path_cost + h

    if f > threshold:
        return None, f, nodes_created

    if graph.is_goal(node.state):
        return node, f, nodes_created
  
    min_threshold = float("inf")
    neighbours = graph.get_neighbours(node.state.id)

    for neighbour, cost in neighbours:
        if neighbour.id in path_ids:
            continue

        child = SearchNode(
            state=neighbour,
            parent=node,
            action=cost,
            path_cost=node.path_cost + cost
        )
        nodes_created += 1

        path_ids.add(neighbour.id)

        result, temp_threshold, nodes_created = search(
            child, graph, goals, threshold, path_ids, nodes_created
        )

        path_ids.remove(neighbour.id)
        if result is not None:
            return result, temp_threshold, nodes_created

        if temp_threshold < min_threshold:
            min_threshold = temp_threshold
    return None, min_threshold, nodes_created
  
def ida_star(graph):
    start = graph.origin
    goals = graph.destinations

    start_node = SearchNode(state=start, parent=None, action=None, path_cost=0)

    threshold = nearest_goal_dist(start, goals)
    nodes_created = 1

    while True:
        path_ids = {start.id}
    
        result, new_threshold, nodes_created = search(
            start_node, graph, goals, threshold, path_ids, nodes_created
        )

        if result is not None:
            return result.state.id, nodes_created, result.get_path()

        if new_threshold == float("inf"):
            return None, nodes_created, []

        threshold = new_threshold
