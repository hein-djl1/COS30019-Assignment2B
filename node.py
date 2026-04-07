class Node:
    """A location in the graph, basically a state"""

    def __init__(self, node_id, x, y):
        self.id = node_id
        self.x = x
        self.y = y
    
    #making pritning the object a bit nicer
    def __repr__(self):
        return f"Node({self.id})"

    def __eq__(self, other):
        return isinstance(other, Node) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


class SearchNode:
    """
    A node in the search tree.

    Attributes
    ----------
    state     : Node   — the graph location this search node represents
    parent    : SearchNode | None  — the node we came from (None at the start)
    action    : float | None       — cost of the edge taken to reach here
    path_cost : float              — total cost from origin to here
    """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state     = state
        self.parent    = parent
        self.action    = action
        self.path_cost = path_cost

    def get_path(self):
        """
        just call this to get the path from the origin to the current node

        Example:
            origin -> 2 -> 3 -> 5   returns [2, 3, 5]
        """
        path = []
        current = self
        
        # keeping going up the parents until we hit the origin (where parent is none)
        while current is not None:
            path.append(current.state.id)
            current = current.parent
        # reverse it so it goes from origin to destination
        path.reverse()
        return path

    #making printing the object a bit nicer
    def __repr__(self):
        parent_id = self.parent.state.id if self.parent else None
        return (f"SearchNode(state={self.state.id}, "
                f"parent={parent_id}, path_cost={self.path_cost})")
