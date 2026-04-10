import sys
from node import SearchNode


def iddfs(graph):

    # to track how many nodes were created 
    total_nodes_created = 0
    start = graph.origin
    
    #depth here is used for depth limit in DLS, we keep increasing it until we find the goal or exhaust the search space
    for depth in range(sys.maxsize):
        #origin node
        start_search_node = SearchNode(state=start, parent=None, action=None, path_cost=0)
        total_nodes_created += 1
        
        result, nodes_created_in_dls = dls(start_search_node, graph, depth)
        total_nodes_created += nodes_created_in_dls
        
        # goal is found, return the result
        if result != 'cutoff' and result is not None:
            return result.state.id, total_nodes_created, result.get_path()
        elif result == 'cutoff':
            # 'cutoff' means we hit the depth limit without finding the goal, so we need to increase the depth and try again
            continue
        # if result is None, the entire search space has been explored, no solution
        elif result is None:
            return None, total_nodes_created, []
            
    return None, total_nodes_created, []


def dls(start_node, graph, limit):
    """
    Iterative Depth-Limited Search using an explicit stack.
    """
    nodes_created = 0
    cutoff_occurred = False
    
    # The stack stores tuples of: (current_node, current_depth, current_path_states)
    stack = [(start_node, 0, {start_node.state.id})]

    while stack:
        # Pop the most recently added item off the stack
        current_node, current_depth, current_path = stack.pop()

        #this line if only for debuggin purposes, it shows the current node being visited, the depth of that node and how many items are waiting in the stack to be visited
        #print(f"Visiting Node: {current_node.state.id} | Current Depth: {current_depth} | Items waiting in Stack: {len(stack)}")

        # Check for the goal
        if graph.is_goal(current_node.state):
            return current_node, nodes_created

        # Check the depth limit
        if current_depth == limit:
            cutoff_occurred = True
            continue # Skip expanding this node, but keep processing the stack

        # Expand neighbours
        neighbours = graph.get_neighbours(current_node.state.id)
        for neighbour, cost in reversed(neighbours):
            # Cycle checking
            if neighbour.id in current_path:
                continue
                
            # Create the child node, path cost is NOT used for IDDFS
            child = SearchNode(
                state=neighbour,
                parent=current_node,
                action=cost,
                path_cost=current_node.path_cost + cost
            )
            nodes_created += 1
            
            # Create a NEW set for this specific branch's path
            new_path = current_path.copy()
            new_path.add(neighbour.id)
            
            # Push the child onto the stack to be evaluated later
            stack.append((child, current_depth + 1, new_path))

    # Final evaluation after the stack is empty
    if cutoff_occurred:
        return 'cutoff', nodes_created
    else:
        return None, nodes_created
