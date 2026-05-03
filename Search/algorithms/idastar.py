from node import SearchNode
from heuristic import nearest_goal_dist

DEBUG = False

def ida_star(graph):
    """Iterative Deepening A* Search Algorithm"""

    # get the starting node and set of goal nodes from the graph
    start = graph.origin
    goals = graph.destinations

    # create the initial search node for the start state
    start_node = SearchNode(state=start, parent=None, action=None, path_cost=0)

    # initial threshold is the heuristic estimate from start to nearest goal
    threshold = nearest_goal_dist(start, goals)
    nodes_created = 1

    if DEBUG:
        print(f"[DEBUG] Start IDA* from node {start.id}")
        print(f"[DEBUG] Goals: {[goal.id for goal in goals]}")
        print(f"[DEBUG] Initial threshold: {threshold}")
        print(f"[DEBUG] Initial nodes_created: {nodes_created}")

    # the main IDA* loop that will repeatedly deepen the search threshold
    while True:
        path_ids = {start.id}

        if DEBUG:
            print(f"\n[DEBUG] --- New IDA* Iteration ---")
            print(f"[DEBUG] Current threshold: {threshold}")
            print(f"[DEBUG] Initial path_ids: {path_ids}")
    
        result, new_threshold, nodes_created = search(
            start_node, graph, goals, threshold, path_ids, nodes_created
        )

        if DEBUG:
            if result is not None:
                print(f"[DEBUG] Solution found!")
                print(f"[DEBUG] Goal node ID: {result.state.id}")
                print(f"[DEBUG] Path: {result.get_path()}")
                print(f"[DEBUG] Total nodes created: {nodes_created}")
            else:
                print(f"[DEBUG] No solution found in this iteration")
                print(f"[DEBUG] New threshold: {new_threshold}")

        #If a solution is found, then return goal ID, node count and path
        if result is not None:
            return result.state.id, nodes_created, result.get_path()

        # If no solution and no more threshold to try, search fails
        if new_threshold == float("inf"):
            if DEBUG:
                print(f"[DEBUG] Search failed - no more thresholds to try")
                print(f"[DEBUG] Total nodes created: {nodes_created}")
            return None, nodes_created, []

        # Otherwise, increase the threshold for next iteration
        threshold = new_threshold
        if DEBUG:
            print(f"[DEBUG] Updating threshold to {threshold} for next iteration")

# Recursive depth-limited search for IDA*
def search(node, graph, goals, threshold, path_ids, nodes_created):
    # Calculate heuristic for current node
    h = nearest_goal_dist(node.state, goals)
    f = node.path_cost + h

    if DEBUG:
        print(f"[DEBUG] Searching node {node.state.id}")
        print(f"[DEBUG]   path_cost={node.path_cost}, h={h}, f={f}, threshold={threshold}")
        print(f"[DEBUG]   Current path: {node.get_path()}")

    if f > threshold:
        if DEBUG:
            print(f"[DEBUG]   PRUNING: f={f} > threshold={threshold}")
            print(f"[DEBUG]   Returning min_threshold={f}")
        return None, f, nodes_created

    if graph.is_goal(node.state):
        if DEBUG:
            print(f"[DEBUG]   GOAL FOUND at node {node.state.id}!")
            print(f"[DEBUG]   Path cost: {node.path_cost}")
        return node, f, nodes_created
  
    min_threshold = float("inf")
    neighbours = graph.get_neighbours(node.state.id)

    if DEBUG:
        print(f"[DEBUG]   Neighbours of {node.state.id}: {[neighbour.id for neighbour, cost in neighbours]}")
        print(f"[DEBUG]   Current path_ids: {path_ids}")

    for neighbour, cost in neighbours:
        if neighbour.id in path_ids:
            if DEBUG:
                print(f"[DEBUG]   Skipping node {neighbour.id} (already in path)")
            continue

        child = SearchNode(
            state=neighbour,
            parent=node,
            action=cost,
            path_cost=node.path_cost + cost
        )
        nodes_created += 1

        if DEBUG:
            print(f"[DEBUG]   Created child node {neighbour.id} (edge cost={cost}, path_cost={child.path_cost})")
            print(f"[DEBUG]   Total nodes created so far: {nodes_created}")

        path_ids.add(neighbour.id)

        if DEBUG:
            print(f"[DEBUG]   Recursing into node {neighbour.id}...")
            print(f"[DEBUG]   Updated path_ids: {path_ids}")

        result, temp_threshold, nodes_created = search(
            child, graph, goals, threshold, path_ids, nodes_created
        )

        path_ids.remove(neighbour.id)

        if DEBUG:
            print(f"[DEBUG]   Backtracked from node {neighbour.id}")
            print(f"[DEBUG]   Restored path_ids: {path_ids}")
            if result is not None:
                print(f"[DEBUG]   Solution found in subtree!")
                
        if result is not None:
            return result, temp_threshold, nodes_created

        if temp_threshold < min_threshold:
            min_threshold = temp_threshold
            if DEBUG:
                print(f"[DEBUG]   Updated min_threshold to {min_threshold}")

    if DEBUG:
        if min_threshold == float("inf"):
            print(f"[DEBUG]   No valid children found - returning failure")
        else:
            print(f"[DEBUG]   Returning min_threshold={min_threshold}")
    return None, min_threshold, nodes_created
  
