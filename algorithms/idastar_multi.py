from node import SearchNode
from heuristic import nearest_goal_dist

DEBUG = False

def ida_star_multi(graph):
    """
    Multi-Destination IDA* Search Algorithm

    Extends IDA* to find the shortest route that visits ALL destination nodes.

    Key modifications from standard IDA*:
    1. Relaxed cycle detection: a node may be revisited if the number of
       remaining (unvisited) destinations has decreased since the last visit.
    2. Goal test: the search only terminates when every destination has been
       visited, not just when any single one is reached.
    3. Dynamic heuristic: at every step the heuristic evaluates the distance
       to the nearest unvisited destination.
    """

    start = graph.origin
    goals = set(graph.destinations)

    start_node = SearchNode(state=start, parent=None, action=None, path_cost=0)

    # initial threshold is the heuristic to the nearest goal from the start
    threshold = nearest_goal_dist(start, goals)
    nodes_created = 1

    # all destinations are initially unvisited
    unvisited = set(goals)

    if DEBUG:
        print(f"[DEBUG] Start Multi-IDA* from node {start.id}")
        print(f"[DEBUG] Goals: {[g.id for g in goals]}")
        print(f"[DEBUG] Initial threshold: {threshold}")

    while True:
        # visited_with_remaining tracks: node_id -> fewest remaining
        # destinations when that node was last visited on the current path.
        # A node can be re-expanded if remaining destinations have shrunk.
        visited_with_remaining = {start.id: len(unvisited)}

        if DEBUG:
            print(f"\n[DEBUG] --- New Multi-IDA* Iteration ---")
            print(f"[DEBUG] Current threshold: {threshold}")

        result, new_threshold, nodes_created = _search(
            start_node, graph, goals, unvisited.copy(),
            threshold, visited_with_remaining, nodes_created
        )

        if result is not None:
            if DEBUG:
                print(f"[DEBUG] Solution found!")
                print(f"[DEBUG] Path: {result.get_path()}")
                print(f"[DEBUG] Total nodes created: {nodes_created}")
            # return all goal ids joined by semicolons, node count, and path
            goal_ids = " ".join(str(g.id) for g in goals)
            return goal_ids, nodes_created, result.get_path()

        if new_threshold == float("inf"):
            if DEBUG:
                print(f"[DEBUG] Search failed - no more thresholds to try")
            return None, nodes_created, []

        threshold = new_threshold
        if DEBUG:
            print(f"[DEBUG] Updating threshold to {threshold}")


def _search(node, graph, all_goals, unvisited, threshold,
            visited_with_remaining, nodes_created):
    """
    Recursive depth-limited search for multi-destination IDA*.

    Parameters
    ----------
    node       : SearchNode – current search node
    graph      : Graph
    all_goals  : set[Node]  – complete set of destination nodes
    unvisited  : set[Node]  – destinations not yet visited on this path
    threshold  : float      – current f-value cutoff
    visited_with_remaining : dict[int, int]
        maps node_id to the smallest 'len(unvisited)' seen when that node
        was expanded on this path; allows revisits only when progress is made.
    nodes_created : int     – running counter of nodes created
    """

    # --- dynamic heuristic: target closest *unvisited* destination ---
    if unvisited:
        h = nearest_goal_dist(node.state, unvisited)
    else:
        h = 0  # all destinations visited; just need to confirm termination

    f = node.path_cost + h

    if DEBUG:
        print(f"[DEBUG] Searching node {node.state.id}")
        print(f"[DEBUG]   path_cost={node.path_cost}, h={h}, f={f}, "
              f"threshold={threshold}")
        print(f"[DEBUG]   Unvisited goals: {[g.id for g in unvisited]}")

    # pruning
    if f > threshold:
        if DEBUG:
            print(f"[DEBUG]   PRUNING: f={f} > threshold={threshold}")
        return None, f, nodes_created

    # --- modified goal test: check if current node is a destination ---
    if node.state in unvisited:
        unvisited.remove(node.state)
        if DEBUG:
            print(f"[DEBUG]   Reached destination {node.state.id}! "
                  f"Remaining: {[g.id for g in unvisited]}")

        # if all destinations have been visited, we are done
        if not unvisited:
            if DEBUG:
                print(f"[DEBUG]   ALL GOALS REACHED!")
            return node, f, nodes_created

    # expand neighbours
    min_threshold = float("inf")
    neighbours = graph.get_neighbours(node.state.id)

    if DEBUG:
        print(f"[DEBUG]   Neighbours of {node.state.id}: "
              f"{[nb.id for nb, _ in neighbours]}")

    for neighbour, cost in neighbours:
        remaining_count = len(unvisited)

        # allow revisit if the number of remaining destinations has
        # decreased since this node was last seen on the current path
        if neighbour.id in visited_with_remaining:
            prev_remaining = visited_with_remaining[neighbour.id]
            if remaining_count >= prev_remaining:
                if DEBUG:
                    print(f"[DEBUG]   Skipping node {neighbour.id} "
                          f"(visited with {prev_remaining} remaining, "
                          f"current {remaining_count})")
                continue

        child = SearchNode(
            state=neighbour,
            parent=node,
            action=cost,
            path_cost=node.path_cost + cost
        )
        nodes_created += 1

        if DEBUG:
            print(f"[DEBUG]   Created child node {neighbour.id} "
                  f"(edge cost={cost}, path_cost={child.path_cost})")

        # record this visit with the current remaining count
        old_remaining = visited_with_remaining.get(neighbour.id, None)
        visited_with_remaining[neighbour.id] = remaining_count

        # save unvisited state for backtracking
        saved_unvisited = unvisited.copy()

        result, temp_threshold, nodes_created = _search(
            child, graph, all_goals, unvisited,
            threshold, visited_with_remaining, nodes_created
        )

        # backtrack: restore unvisited set and visited_with_remaining
        unvisited.clear()
        unvisited.update(saved_unvisited)

        if old_remaining is None:
            del visited_with_remaining[neighbour.id]
        else:
            visited_with_remaining[neighbour.id] = old_remaining

        if DEBUG:
            print(f"[DEBUG]   Backtracked from node {neighbour.id}")

        if result is not None:
            return result, temp_threshold, nodes_created

        if temp_threshold < min_threshold:
            min_threshold = temp_threshold

    # backtrack: re-add current node to unvisited if it's a goal
    # (this is needed because we removed it above when we "visited" it)
    if node.state in all_goals and node.state not in unvisited:
        unvisited.add(node.state)
        if DEBUG:
            print(f"[DEBUG]   Restored {node.state.id} to unvisited on backtrack")

    return None, min_threshold, nodes_created
