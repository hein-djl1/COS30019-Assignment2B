from node import SearchNode
from collections import deque

DEBUG = False

def bfs(graph):
    """Breadth First Search"""
    # get the starting node from the graph
    start = graph.origin
    # create the first search node (root of the search tree)
    start_node = SearchNode(state=start, parent=None, action=None, path_cost=0)
    
    # BFS uses a queue (FIFO), so start by putting the origin in it
    # Using deque for efficient pop from left (O(1) instead of O(n) for list)
    queue = deque([start_node])

    # keep track of nodes that have already been added
    # this helps prevent revisiting the same node again
    seen = {start.id}

    # count how many search nodes have been created
    nodes_created = 1

    if DEBUG:
        print(f"[DEBUG] Start BFS from node {start.id}")
        print(f"[DEBUG] Initial queue: {[node.state.id for node in queue]}")
        print(f"[DEBUG] Seen set: {seen}")

    # keep searching until there are no more nodes left to explore
    while queue:
        if DEBUG:
            print("\n[DEBUG] --- New Iteration ---")
            print(f"[DEBUG] Queue before expansion: {[node.state.id for node in queue]}")
        
        # BFS: take the FIRST node added to the queue (FIFO)
        current = queue.popleft()

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

        # For BFS, process neighbours in ascending order (smaller node IDs first)
        # Since queue is FIFO, don't need to reverse, just sort normally
        # But need to maintain the tie-breaking rule: smaller node IDs expanded first
        neighbours_sorted = sorted(neighbours, key=lambda x: x[0].id)
        
        for neighbour, cost in neighbours_sorted:
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

            # BFS: add the child to the END of the queue
            queue.append(child)
            # count this newly created search node
            nodes_created += 1
            
            if DEBUG:
                print(
                    f"[DEBUG] Added node {neighbour.id} to queue "
                    f"(edge cost={cost}, path_cost={child.path_cost})"
                )
        
        if DEBUG:
            print(f"[DEBUG] Queue after expansion: {[node.state.id for node in queue]}")
            print(f"[DEBUG] Seen set now: {seen}")
                
    # if queue becomes empty and no goal was found, return failure
    if DEBUG:
        print("[DEBUG] No path found")
        print(f"[DEBUG] Total nodes created: {nodes_created}")
    return None, nodes_created, []
