
from node import Node


class Graph:
    def __init__(self):
        #for storing graph nodes
        self.nodes = {}
        # node_id -> list of (neighbour_id, cost)
        self.edges = {}
        #origin node
        self.origin = None
        #destination nodes
        self.destinations = []

    def add_node(self, node_id, x, y):
        node = Node(node_id, x, y)
        self.nodes[node_id] = node
        self.edges[node_id] = []
        return node

    def add_edge(self, from_id, to_id, cost):
        self.edges[from_id].append((to_id, cost))

    def get_neighbours(self, node_id):
        def get_id(edge):
            #arrange it using destination node id
            return edge[0]
            
        neighbours = self.edges[node_id]
        #neighbours will look something like (2, 4) which means to node 2 with cost 4
        #sort it using ids
        sorted_neighbours = sorted(neighbours, key=get_id)
        
        result = []
        for node_id, cost in sorted_neighbours:
            result.append((self.nodes[node_id], cost))
        return result

    def is_goal(self, node):
        return node in self.destinations
