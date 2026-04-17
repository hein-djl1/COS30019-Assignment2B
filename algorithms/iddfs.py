import sys
from node import SearchNode

#turn this to true only for debugging
DEBUG = False

def iddfs(graph):
    """Iterative Deepening DFS"""
    total_nodes = 0
    start = graph.origin
    
    if DEBUG:
        print(f"[DEBUG] Starting IDDFS from origin: {start.id}")
    
    # Increase depth one level at a time: 0, 1, 2, ... until we find a goal
    # or prove there is no reachable goal.
    for limit in range(sys.maxsize):
        if DEBUG:
            print(f"\n[DEBUG] Starting iteration with Depth Limit: {limit} ---")
            
        # Each iteration starts fresh from the root with the current depth cap.
        start_node = SearchNode(start, None, None, 0)
        total_nodes += 1
        
        # Stack items are (node, depth, path_ids_set).
        # The path set is per-branch, so cycle checks only block loops in
        # the current path (not valid revisits from other branches).
        stack = [(start_node, 0, {start.id})]
        cutoff_occurred = False
        goal_node = None
        
        while stack:
            current, depth, current_path = stack.pop()
            
            if DEBUG:
                print(f"[DEBUG] Visiting node: {current.state.id} at depth {depth} (Current Path: {[n for n in current_path]})")
            
            if graph.is_goal(current.state):
                if DEBUG:
                    print(f"[DEBUG] Goal '{current.state.id}' found at depth {depth} ***")
                goal_node = current
                break
                
            # Reaching the current depth limit is not failure; it means
            # "try again with a larger limit".
            if depth == limit:
                if DEBUG:
                    print(f"[DEBUG] Cutoff reached at node {current.state.id}")
                cutoff_occurred = True
                continue
                
            neighbours = graph.get_neighbours(current.state.id)
            if DEBUG and neighbours:
                print(f"[DEBUG] Expanding node {current.state.id}...")
                
            # Reverse so the left-to-right neighbor order is preserved under
            # LIFO stack behavior (classic DFS expansion order).
            for neighbour, cost in reversed(neighbours):
                if neighbour.id not in current_path:
                    child = SearchNode(neighbour, current, cost, current.path_cost + cost)
                    total_nodes += 1
                    
                    # Copy path state so each pushed branch tracks its own
                    # visited-on-path set independently.
                    new_path = current_path.copy()
                    new_path.add(neighbour.id)
                    stack.append((child, depth + 1, new_path))
                    
                    if DEBUG:
                        print(f"[DEBUG] Pushed neighbour '{neighbour.id}' to stack (Target depth: {depth + 1})")
                else:
                    if DEBUG:
                        print(f"[DEBUG] Skipped neighbour '{neighbour.id}' (Already in current path)")
                    
        if goal_node:
            return goal_node.state.id, total_nodes, goal_node.get_path()
        # No cutoff means we fully explored everything up to this frontier,
        # so there is nothing deeper left to try.
        if not cutoff_occurred:
            if DEBUG:
                print("[DEBUG] Search space completely exhausted. No goal found.")
            return None, total_nodes, []
            
    return None, total_nodes, []