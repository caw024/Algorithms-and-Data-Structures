# using dp to compute info about trees
# i.e. number of nodes below subtree

'''
def diameter(node: TreeNode) -> int:
    # Algo 1: pick root and do dp (O(n))
    # answer = max(toLeaf(node_i) + toLeaf(node_j)) 
    # where node_i, node_j are children of node
    # toLeaf is max distance from node to leaf
    # max diameter passes through at least one of these nodes

    d = sorted( [toLeaf(child) for child in node.children], reverse=True)
    ans = max(ans, d[0] + d[1]) if len(d) > 1 else max(ans, d[0])
    for child in node.children:
        diamter(child)

    # Algo 2: two dfs
    # given point A, find a farthest point from A, say B.
    # then find farthest point from B, say C.
    # distance from B,C is max distance (if not, contradicts)

'''

class DSU:
    def __init__(self, N):
        self.p = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        if x != self.p[x]: self.p[x] = self.find(self.p[x])
        return self.p[x]

    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        if xr == yr: return False
        if self.sz[xr] > self.sz[yr]: xr, yr = yr, xr
        self.p[xr], self.sz[yr] = yr, self.sz[yr] + self.sz[xr]
        return True
        
    def isConnected(self,x,y):
        return self.find(x) == self.find(y)
    def getSize(self, x):
        return self.sz[self.find(x)]
    def numRoots(self):
        return len(set(self.find(x) for x in range(len(self.p))))

def kruskal(d):
    ''' Constructs min spanning tree in ElogV time '''
    # say that d maps v1 to (v2, e)
    mst_weight = 0
    dsu = DSU(len(d))

    edge_list = [(v[1], k, v[0]) for k,v in d.items()]
    edge_list.sort()
    for e, v1, v2 in edge_list:
        if dsu.find(v1) == dsu.find(v2):
            continue
        mst_weight += e
        dsu.union(v1, v2)
    return mst_weight


from heapq import heappush, heappop
def prims(d):
    ''' Constructs min spanning tree in ElogV time
    Can be improved to E + VlogV through Fibonacci heap '''
    # say we modify d so it maps v1 to (e, v2)
    mst_weight = 0

    v0 = list(d.keys())[0]
    pq = d[v0].copy()
    visited = {v0}
    while pq:
        e, v = heappop(pq)
        if v in visited:
            continue
        visited.add(v)
        for new_edge in d[v]:
            heappush(pq, new_edge)
        mst_weight += e
    return mst_weight


def find_tree_diameter(d):
    """ find tree diameter (or longest path) and then find the center point (or root)"""

    # dfs from 1, storing the nodes with max distance
    def _dfs(node, dist, visited):
        # can pass it in bc set is pass by object reference
        if node in visited:
            return
        visited[node] = dist
        for nbr in d[node]:
            _dfs(nbr, dist+1, visited)

    # also keeps the path formation
    def _path_between_vertices(cur_node, end_node, cur_path):
        if cur_node == end_node:
            nonlocal final_path
            cur_path.append(end_node)
            final_path = cur_path.copy()
            cur_path.pop()
            return

        cur_path.append(cur_node)
        for nbr in d[cur_node]:
            if len(cur_path) > 1 and nbr == cur_path[-2]:
                continue
            _path_between_vertices(nbr, end_node, cur_path)
        # remove most recent one
        cur_path.pop()
    
    visited = {}
    _dfs(1, 0, visited)
    max_val = max(visited.values())
    node_b = [k for k in visited if visited[k] == max_val][0]

    visited = {}
    _dfs(node_b, 0, visited)
    max_val = max(visited.values())
    node_c = [k for k in visited if visited[k] == max_val][0]

    final_path = []
    _path_between_vertices(node_b, node_c, [])
    return final_path