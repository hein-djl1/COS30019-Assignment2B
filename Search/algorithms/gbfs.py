from queue import Empty
import heapq
from node import SearchNode
from heuristic import nearest_goal_dist


def gbfs(graph):

    start = graph.origin
    goals = graph.destinations

    #state is just coordinates
    #parent is where the node came from
    #action is the cost of the edge taken to reach here
    #path_cost is the cumulative cost from origin
    start_search_node = SearchNode(state=start, parent=None, action=None, path_cost=0)

    # seen holds graph node IDs already added to the frontier,
    # so we don't add duplicate entries for the same location.
    seen = {start.id}

    h_start = nearest_goal_dist(start, goals)
    frontier = []
    # basically treat frontier as priority queue
    # h_start or first value is the heuristic value which is what it will be arragned by
    heapq.heappush(frontier, (h_start, start.id, start_search_node))

    # nodes created (each push to the frontier = one new search node)
    nodes_created = 1

    while frontier:

        # debugging lines to show the frontier before popping
        #print(f"Frontier before popping: {[(item[0], item[1]) for item in frontier]}")  
        
        h_val, _, current = heapq.heappop(frontier)

        # Debugging line to show the current node being expanded and the contents of the frontier
        # Create a clean list of what is waiting in the queue: (heuristic, node_id)
        #waiting_nodes = [(item[0], item[1]) for item in frontier]
        
        #print(f"Expanding Node: {current.state.id} (h={h_val:.1f}) | Frontier holds: {waiting_nodes}")

        if graph.is_goal(current.state):
            return current.state.id, nodes_created, current.get_path()

        for neighbour, cost in graph.get_neighbours(current.state.id):
            if neighbour.id in seen:
                continue
            seen.add(neighbour.id)
            child = SearchNode(
                state=neighbour,
                parent=current,
                action=cost,
                path_cost=current.path_cost + cost
            )
            h = nearest_goal_dist(neighbour, goals)
            heapq.heappush(frontier, (h, neighbour.id, child))
            nodes_created += 1

    # Frontier empty no path to any goal
    return None, nodes_created, []
