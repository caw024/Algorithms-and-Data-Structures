from heapq import heapify, heappush, heappop
from collections import defaultdict, deque
from math import inf

# adjancency list: a convenient representation of graphs
# graph[v] = neighbors of v
graph = defaultdict(list)

# adjacency_matrix[(u,v)] = length of edge u,v
adjacency_matrix = defaultdict(int)

# edge_list = [ (u,v,dist(u,v)) ]
edge_list = []

# set of all visited nodes
visited = set()

# O(V+E)
def dfs(node):
    global visited

    if node in visited:
        return
    visited.add(node)
    for neighbor in graph[node]:
        dfs(neighbor)
    
# O(V+E)
def bfs(node):
    visited = set()
    visited.add(node)

    q = deque()
    q.append(node)

    while q:
        cur = q.popleft()
        for neighbor in graph[cur]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            q.append(neighbor)


# O(V + ElogE)
# finds shortest path from start node to all other nodes
def dijkstra(node):
    global graph, adjacency_matrix

    visited = set()
    visited.add(node)
    distance = [inf]*len(graph)
    
    min_heap = []
    heappush(min_heap,(0,node))

    while min_heap:
        dist, cur = heappop(min_heap)
        distance[cur] = dist
        
        for neighbor in graph[cur]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            heappush(min_heap, 
                    (adjacency_matrix[(cur, neighbor)] + dist, neighbor)
                    )
    

# O(VE)
# inferior dijkstra but can deal w negative distances
# (note neg cycle = no min dist)
# SFPA is faster version by using queue
def bellman_ford(node):
    global graph, edge_list
    distance = [inf]*len(graph)

    distance[node] = 0
    for _ in range(len(graph)):
        for start, end, dist in edge_list:
            distance[end] = min(distance[end], 
                                distance[start] + dist
                                )
    

# O(V^3)
# gets shortest distance between any two pts
def floyd_warshall():
    global graph, edge_list
    distance = [ [inf]*len(graph) for _ in range(len(graph))]
    for i in range(len(graph)):
        distance[i][i] = 0
    for start, end, dist in edge_list:
        distance[start][end] = dist

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                distance[i][j] = min(distance[i][j],
                                     distance[i][k] + distance[k][j]
                                     )
                

if __name__ == "__main__":
    pass
