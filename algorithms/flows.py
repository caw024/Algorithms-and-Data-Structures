n = 20 # number of edges
# stores the max capacity of directed edges
capacity = [[0]*n for _ in range(n)]

# adj_list should hold ALL UNDIRECTED EDGES
adj_list = [[] for _ in range(n)]
for i in range(n):
    for j in range(n):
        if (capacity[i][j] == 0 and capacity[j][i] == 0):
            continue
        adj_list[i].append(j)

s = 0 # source
t = 1 # sink

# from pprint import pprint
# print(f"{n=}")
# print("capacity")
# pprint(capacity)
# print("adj_list")
# pprint(adj_list)

from collections import deque
from math import inf

def bfs(s: int, t: int) -> int:
    global parent
    parent = [-1] * n
    parent[s] = -2
    q = deque()
    q.append((s, inf))
    while q:
        cur, flow = q.popleft()
        for next_ in adj_list[cur]:
            if (parent[next_] == -1 and capacity[cur][next_]):
                parent[next_] = cur
                new_flow = min(flow, capacity[cur][next_])
                if (next_ == t):
                    return new_flow
                q.append((next_, new_flow))


# computes min-cut = max-flow in O(V * E^2)
def edmonds_karp(s: int, t: int) -> int:
    global parent

    flow = 0
    new_flow = bfs(s, t)
    while (new_flow):
        flow += new_flow
        cur = t
        while (cur != s):
            prev = parent[cur]
            capacity[prev][cur] -= new_flow
            capacity[cur][prev] += new_flow
            cur = prev
        new_flow = bfs(s, t)

    return flow

print(edmonds_karp(s, t))
# Implemented from CP Algorithms
# Well tested in Picture Day Problem