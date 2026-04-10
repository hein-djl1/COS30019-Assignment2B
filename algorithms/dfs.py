from node import SearchNode


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

    # keep searching until there are no more nodes left to explore
    while stack:
        # take the most recently added node front he stack
        current = stack.pop()

        # if current node is one of the destination nodes, return the result
        if graph.is_goal(current.state):
            return current.state.id, nodes_created, current.get_path()

        # get the neighbours of the current node
        neighbours = graph.get_neighbours(current.state.id)

        # reverse the neighbours before pushing to the stack
        # becuase DFS is LIFO, this makes smaller node IDs expanded first
        for neighbour, cost in reversed(neighbours):
            if neighbour.id in seen:
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
    # if stack becomes empty and no goal ewas found, return failure
    return None, nodes_created, []
