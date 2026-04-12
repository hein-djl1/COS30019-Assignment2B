from node import SearchNode

DEBUG = False

def dfs(graph):
    # get the starting node from the graph
    start = graph.origin
    # create the first search node (root of the search tree)
    start_node = SearchNode(state=start, parent=None, action=None, path_cost=0)
    
    # DFS uses a stack, so start bu putting the origin in it
    stack = [start_node]

    # keep track of nodes that have already been added
    # this helps prevent revisiting the same node again
    seen = {start.id}

    # count on how many search nodes have been created
    nodes_created = 1

    if DEBUG:
        print(f"[DEBUG] Start DFS from node {start.id}")
        print(f"[DEBUG] Initial stack: {[node.state.id for node in stack]}")
        print(f"[DEBUG] Seen set: {seen}")

    # keep searching until there are no more nodes left to explore
    while stack:
        if DEBUG:
            print("\n[DEBUG] --- New Iteration ---")
            print(f"[DEBUG] Stack before explored: {[node.state.id for node in stack]}")
        # take the most recently added node front he stack
        current = stack.pop()

        if DEBUG:
            print(f"[DEBUG] Exploring node {current.state.id}")
            print(f"[DEBUG] Current path: {current.get_path()}")
        # if current node is one of the destination nodes, return the result
        if graph.is_goal(current.state):
            if DEBUG:
                print(f"[DEBUG] Goal found at node {current.state.id}")
                print(f"[DEBUG] Final path: {current.get_path()}")
                print(f"[DEBUG] Total nodes created: {nodes_created}")
            return current.state.id, nodes_created, current.get_path()

        # get the neighbours of the current node
        neighbours = graph.get_neighbours(current.state.id)
        if DEBUG:
            print(f"[DEBUG] Neighbours of {current.state.id}: {[neighbour.id for neighbour, cost in neighbours]}")

        # reverse the neighbours before pushing to the stack
        # becuase DFS is LIFO, this makes smaller node IDs expanded first
        for neighbour, cost in reversed(neighbours):
            if neighbour.id in seen:
                if DEBUG:
                    print(f"[DEBUG] Skip node {neighbour.id} because it is already seen")
                continue

            # mark this neighbour as seen
            seen.add(neighbour.id)

            # create a new search node for this neighbour
            child = SearchNode(
                state=neighbour,
                parent=current,
                action=cost,
                path_cost=current.path_cost + cost
            )

            # add the child to the stack so DFS can explore it later
            stack.append(child)
            # count this newly created search node
            nodes_created += 1
            if DEBUG:
                print(
                    f"[DEBUG] Added node {neighbour.id} to stack "
                    f"(edge cost={cost}, path_cost={child.path_cost})"
                )
            if DEBUG:
                print(f"[DEBUG] Stack after expansion: {[node.state.id for node in stack]}")
                print(f"[DEBUG] Seen set now: {seen}")
                
    # if stack becomes empty and no goal ewas found, return failure
    if DEBUG:
        print("[DEBUG] No path found")
        print(f"[DEBUG] Total nodes created: {nodes_created}")
    return None, nodes_created, []
