import heapq
from node import SearchNode
from heuristic import nearest_goal_dist


def astar(graph):
    """
    A* Search Algorithm
    
    Evaluates nodes using f(n) = g(n) + h(n) where:
        g(n) = cost to reach current node from origin
        h(n) = estimated cost to nearest goal (Euclidean distance)
    
    Tie-breaking: when f values are equal, expand smaller node id first.
    When node ids are also equal (different branches), expand the one added
    earlier (chronological order) — handled naturally by the counter.

    Returns:
        (goal_id, nodes_created, path)
        goal_id      : int   — id of the goal node reached
        nodes_created: int   — total SearchNodes created during search
        path         : list  — list of node ids from origin to goal
    """

    origin = graph.origin
    goals  = graph.destinations

    # Each heap entry: (f, counter, search_node)
    # counter breaks ties chronologically — earlier added = smaller counter
    counter      = 0
    nodes_created = 0

    start_node = SearchNode(state=origin, parent=None, action=None, path_cost=0)
    nodes_created += 1

    h_start = nearest_goal_dist(origin, goals)
    f_start = 0 + h_start  # g=0 at origin

    # min-heap
    frontier = []
    heapq.heappush(frontier, (f_start, counter, start_node))
    counter += 1

    # Track the best g-cost seen for each state to avoid redundant expansions.
    # (A* on a tree still benefits from this when the same state can be reached
    # via multiple paths — it avoids re-expanding states with worse g values.)
    best_g = {origin: 0}

    while frontier:
        f, _, current = heapq.heappop(frontier)

        current_state = current.state

        # Goal check on expansion (not on generation) — standard A*
        if graph.is_goal(current_state):
            return current_state.id, nodes_created, current.get_path()

        # Skip if we have already found a cheaper path to this state
        if current.path_cost > best_g.get(current_state, float('inf')):
            continue

        # Expand neighbours — graph.get_neighbours already sorts by node id
        for neighbour, cost in graph.get_neighbours(current_state.id):
            new_g = current.path_cost + cost

            # Only proceed if this path is strictly cheaper than any seen before
            if new_g < best_g.get(neighbour, float('inf')):
                best_g[neighbour] = new_g

                h = nearest_goal_dist(neighbour, goals)
                f_new = new_g + h

                child = SearchNode(
                    state=neighbour,
                    parent=current,
                    action=cost,
                    path_cost=new_g
                )
                nodes_created += 1

                # Tie-break on node id (ascending), then counter (chronological)
                heapq.heappush(frontier, (f_new, neighbour.id, child))
                counter += 1

    # No path found
    return None, nodes_created, []