import math


def euclidean(node, goal):
    """Straight line distance between two nodes."""
    return math.sqrt((node.x - goal.x) ** 2 + (node.y - goal.y) ** 2)


def nearest_goal_dist(node, goals):
    """
    Return the Euclidean distance from node to the nearest goal.
    Used when there are multiple destination nodes.
    """
    return min(euclidean(node, g) for g in goals)
