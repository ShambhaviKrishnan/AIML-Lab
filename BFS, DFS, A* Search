from collections import deque
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# ---------------- Graph Data ----------------
graph = {
    'A': ['B', 'C'],
    'B': ['D'],
    'C': ['E'],
    'D': ['E'],
    'E': []
}

heuristic = {
    'A': 3, 'B': 2, 'C': 2, 'D': 1, 'E': 0
}

# ---------------- Search Algorithms ----------------

def bfs(start, goal):
    queue = deque([[start]])
    visited = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == goal: return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                queue.append(path + [neighbor])
    return None

def dfs(start, goal):
    stack = [[start]]
    visited = set()
    while stack:
        path = stack.pop()
        node = path[-1]
        if node == goal: return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                stack.append(path + [neighbor])
    return None

def astar(start, goal):
    pq = []
    heapq.heappush(pq, (heuristic[start], 0, [start]))
    visited = set()
    while pq:
        f, cost, path = heapq.heappop(pq)
        node = path[-1]
        if node == goal: return path
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                new_cost = cost + 1
                f_value = new_cost + heuristic[neighbor]
                heapq.heappush(pq, (f_value, new_cost, path + [neighbor]))
    return None


def visualize_graph(graph_dict, path=None, title="Graph Representation"):
    G = nx.DiGraph()
    for node, neighbors in graph_dict.items():
        for neighbor in neighbors:
            G.add_edge(node, neighbor)

    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 5))

    # Draw all nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=1500, font_size=15, arrowsize=20)

    # Highlight the path if provided
    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='orange')
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title(title)
    plt.show()

start_node = 'A'
goal_node = 'E'

path_bfs = bfs(start_node, goal_node)
path_dfs = dfs(start_node, goal_node)
path_astar = astar(start_node, goal_node)

print("BFS Path:", path_bfs)
print("DFS Path:", path_dfs)
print("A* Path :", path_astar)

# Call the visualization for the A* result
visualize_graph(graph, path=path_astar, title="A* Optimal Path Highlighted")
