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


### EFFICIENT TREE QUERIES


''' Find kth ancestor of a node in logarithmic time per query 
nlogn preprocessing to generate ancestor(x,2^i)'''
def ancestors(nodes: list) -> dict:
    # nodes should be list of nodes with distinct values (or ids) and access to parent node
    ancestors = {}
    for node in nodes:
        ancestors[(node.val, 1)] = node.parent.val

    i = 1
    while True:
        if (i << 1) >= len(nodes):
            break
        prev_node_id = ancestors[(node.val, i)]
        ancestors[(node.val, i << 1)] = ancestors[(prev_node_id, i)]
        i <<= 1
    return ancestors

def assign_graph_ids_via_dfs(node: int, lru_id: int, visited: set, 
                                    node_ids: dict, neighbors: dict) -> int:
    if node in visited:
        return lru_id
    visited.add(node)
    lru_id += 1
    node_ids[node] = lru_id
    for nbr in neighbors[node]:
        lru_id = assign_graph_ids_via_dfs(nbr, lru_id+1, visited)
    return lru_id

def compute_subtree_sizes(node: int, subtree_sizes: dict, neighbors: dict) -> int:
    cur_subtree_size = 1
    for nbr in neighbors[node]:
        cur_subtree_size += compute_subtree_sizes(nbr, subtree_sizes, neighbors)
    subtree_sizes[node] = cur_subtree_size
    return cur_subtree_size

''' Calculating dynamic subtree queries in O(logn)'''
# BUILD A TREE TRAVERSAL ARRAY
# store values of nodes in segment tree, can update value and calculate sum in O(logn)

''' Calculating dynamic path queries in O(logn) '''
# similar to before, but need to change values of list in a range, logn using segment tree

''' Lowest common ancestor of node query in O(logn) '''
'''
Approach 1: 
Compute level of each node in O(n) using DFS
Move lower pointer node to the same level as the other node (using ancestors)
Binary search for the 1st time both pointers share an ancestor
- this step is actually O(logn) because each ancestor(x,2**n) is computed in O(1)
- i.e. find largest n s.t. ancestor(A,2**n) != ancestor(B,2**n) then work up
O(logn) per query
'''

'''
Approach 2:
Use a different tree traversal array generated from DFS
- add node whenever it appears in dfs tree (forward and backwards)
find min depth between nodes a and b in array, this is a range min query
Pre-process O(nlogn) using range min query
O(1) per query
'''

'''
Approach 3: use offline algo by combining using union find?
'''


'''
Get distance of 2 nodes in a tree in O(logn) time

dist(a,b) = depth(a) + depth(b) - 2 * depth(lca(a,b))
'''