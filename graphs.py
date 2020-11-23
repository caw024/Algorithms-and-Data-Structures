import heapq
from collections import defaultdict, deque

graph = defaultdict(int)
visited = set()

# O(V+E)
def dfs(node):
    nonlocal graph, visited
    if node in visited:
        return
    visited.add(node)
    
    for neighbor in graph[node]:
        dfs(neighbor)
    
# O(V+E)
def bfs(node):
    nonlocal graph
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
    nonlocal graph, edgelen
    visited = set()
    visited.add(node)
    distance = [math.inf for _ in range(len(graph))]
    
    h = []
    heappush(h,(0,node))
    

    while h:
        dist, cur = heappop(h)
        distance[cur] = dist
        
        for neighbor in graph[cur]:
            if neighbor in visited:
                continue
            visited.add(neighbor)
            heappush(h,(edgelen[cur, neighbor] + dist ,neighbor))
    

# O(VE)
# inferior dijkstra but can deal w negative distances
# (note neg cycle = no min dist)
# SFPA is faster version by using queue
def bellmanford(node):
    nonlocal graph, edgelen
    distance = [math.inf]*len(graph)
    h = []

    distance[node] = 0
    for i in range(len(graph)):
        for e in edgelen:
            start, end, dist = e
            distance[end] = min(distance[end], distance[start] + dist)
    

# O(V^3)
# gets shortest distance between any two pts
def floydwarshall():
    nonlocal graph, edgelen
    distance = [ [math.inf]*len(graph) for _ in range(len(graph))]
    for i in range(len(graph)):
        distance[i][i] = 0
    for start, end, dist in edgelen:
        distance[start][end] = dist

    for k in range(len(graph)):
        for i in range(len(graph)):
            for j in range(len(graph)):
                distance[i][j] = min(distance[i][j],
                                     distance[i][k] + distance[k][j])
                

def main():
    pass

if __name__ == "__main__":
    main()
