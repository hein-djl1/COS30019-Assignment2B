from multiprocessing.util import debug
import sys
from node import SearchNode
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
    
    for limit in range(sys.maxsize):
        if DEBUG:
            print(f"\n[DEBUG] Starting iteration with Depth Limit: {limit} ---")
            
        start_node = SearchNode(start, None, None, 0)
        total_nodes += 1
        
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
                
            if depth == limit:
                if DEBUG:
                    print(f"[DEBUG] Cutoff reached at node {current.state.id}")
                cutoff_occurred = True
                continue
                
            neighbours = graph.get_neighbours(current.state.id)
            if DEBUG and neighbours:
                print(f"[DEBUG] Expanding node {current.state.id}...")
                
            for neighbour, cost in reversed(neighbours):
                if neighbour.id not in current_path:
                    child = SearchNode(neighbour, current, cost, current.path_cost + cost)
                    total_nodes += 1
                    
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
        if not cutoff_occurred:
            if DEBUG:
                print("[DEBUG] Search space completely exhausted. No goal found.")
            return None, total_nodes, [] # Search space exhausted
            
    return None, total_nodes, []