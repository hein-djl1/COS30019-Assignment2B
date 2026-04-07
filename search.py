import sys
from parser import parse_file

from algorithms.gbfs import gbfs
# from algorithms.dfs  import dfs
# from algorithms.bfs  import bfs
# from algorithms.astar import astar
# from algorithms.cus1 import cus1
# from algorithms.cus2 import cus2

algorithms = {
    "GBFS": gbfs,
    # "DFS":  dfs,
    # "BFS":  bfs,
    # "AS":   astar,
    # "CUS1": cus1,
    # "CUS2": cus2,
}


def main():
    if len(sys.argv) != 3:
        print("Usage: python search.py <filename> <method>")
        print("Supported methods:", ", ".join(algorithms))
        sys.exit(1)

    filename = sys.argv[1]
    method   = sys.argv[2].upper()

    if method not in algorithms:
        print(f"Method '{method}' not recognised or not yet implemented.")
        print("Supported methods:", ", ".join(algorithms))
        sys.exit(1)

    graph = parse_file(filename)
    goal, nodes_created, path = algorithms[method](graph)

    print(filename, method)
    if goal is None:
        print("No path found.")
    else:
        print(goal, nodes_created)
        print(" -> ".join(str(n) for n in path))


if __name__ == "__main__":
    main()
