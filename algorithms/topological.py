from collections import defaultdict, deque
    
'''
    Returns topologically sort list of nodes (if cycle, dealt with),
    Parameters: (dict) graph expressed as an adjacency list
    Return: (list) topologically sorted graph
    Runtime O(V+E), space O(V)

    Side notes: 
    - capable of dealing w cycles
    - need to deal w self loops and multiple edges between two pts
    - note dictionary graph should include ALL nodes 
            (or else dictionary size changed during iteration)
            might have to replace graph.keys()
        i.e graph.keys() might change, but not graph[node] (directed pointing to vertex)
    - If you want lexographic order; sort the keys and values in graph

'''
def top_sort_dfs(graph):
    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                dfs(neighbor)
        ans.append(node)

    visited = {node: False for node in graph.keys()}
    ans = []
    for node in graph.keys():
        if not visited[node]: dfs(node)

    ans = ans[::-1]
    pos = {node: idx for idx, node in enumerate(ans)}
    bad = any(pos[node] > pos[neighbor] for node in graph.keys() for neighbor in graph[node])
    # O(V + E) again to check if legitimate top sort
    return [] if bad else ans


# make sure to include all nodes in graph 
def get_indegrees(graph):
    indegree = {node: 0 for node in graph.keys()}
    for nodes in graph.values():
        for node in nodes:
            indegree[node] += 1
    return indegree

# kahn's algorithm
def top_sort_bfs(graph):
    order = []
    indegree = get_indegrees(graph)
    q = deque( node for node in indegree if indegree[node] == 0 )
    while q:
        node = q.popleft()
        order.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)

    # same as having idx to show not all vertices were covered
    return [] if len(order) != len(graph.keys()) else order


# O(V + E) dfs until we hit a leaf, add the leaf
def topological(graph):
    visited = {node: 0 for node in d.keys()}
    ans = deque()
    # STATES OF VISITS:
    # 0 = not visited, 1 = still visiting, 2 = completely visited
    def dfs(node):
        if visited[node] == 2:
            return True
        # this gives us a cycle: bad
        elif visited[node] == 1:
            return False
        # node not visited yet
        visited[node] = 1
        if any(not dfs(neighbor) for neighbor in graph[node]):
            return False
        visited[node] = 2
        ans.appendleft(node)
        return True
    return [] if any(not dfs(node) for node in graph.keys()) else ans


'''
Applying DP to DAGs to solve
    1) from points A to B, we can find (using recursion + memoization):
        - number of directed paths 
        - number of shortest/longest paths (modified Dijkstra)
    2) shortest/longest path in entire graph
    3) min/max num edges in path
    4) which node appears in any path

Any DP problem can be expressed as DAG!
'''
# uses recursion: O(V+E)
# can be modified to find longest/shortest path from A to B
def num_paths(graph: dict, A: int, B: int) -> int:
    inverted = defaultdict(list)
    for node in graph.keys():
        for nextnode in graph[node]:
            inverted[nextnode].append(node)
    topsort = top_sort_bfs(graph)
    position = {node: idx for idx, node in enumerate(topsort)}

    # num of paths from A to X; only needs O(V) states
    from functools import lru_cache
    @lru_cache(None)
    def paths(X):
        if X == A:
            return 1
        if position[X] < position[A]:
            return 0
        return sum(paths(prevnode) for prevnode in inverted[X])
    return paths(B)

'''
successor graph: outdegree of each node is exactly 1
    - succ(x,1) is a function of x that tells what next step is
    - let succ(x,k) be k steps from starting point x
        - can be computed in O(logk) per query using sparse table method
        - compute succ(x,1), succ(x,2), succ(x,4)... VlogV preprocessing
            - note succ(x,2^4) = succ( succ(x,2^3), 2^3 )
        - then succ(x,k) = composition of succ(2**i)

cycle detection (which belongs at end) of successor graph in O(n) time, O(1) space
Floyd's algorithm
'''


# APPLICATIONS:
'''
help determine a valid order to perform activities
can check for cycles in directed acyclic graph
'''
if __name__ == "__main__":
    # note d.clear() only needed when other variable references graph
    # d is a directed graph
    # tested against leetcode and codeforces cases
    d = defaultdict(list)
    d = {0: [1,4], 1: [2,3], 2: [3], 3: [4], 4: []}
    print(top_sort_dfs(d))
    print(top_sort_bfs(d))
    print(topological(d))
    # expected [0,1,2,3,4] or some variant

    d = defaultdict(list)
    d= {0:[1,2], 1:[0,2], 2:[], 3:[0]}
    print(top_sort_dfs(d))
    print(top_sort_bfs(d))
    print(topological(d))
    # [] bc a cycle exists

    d = defaultdict(list)
    d = {1:[9,3], 2:[1,3,9], 3:[5,7], 4:[], 5:[4], 6:[1], 7:[9], 8:[4,3], 9:[10], 10:[11], 11:[]}
    print(top_sort_dfs(d))
    print(top_sort_bfs(d))
    print(topological(d))

    # several examples are possible...
    # ie [8, 6, 2, 1, 3, 7, 5, 4, 9, 10, 11]
    # [2, 6, 8, 1, 3, 5, 7, 4, 9, 10, 11]

    d = defaultdict(list)
    d = {1:[2,4], 2:[3], 3:[6], 4:[5], 5:[2,3], 6:[]}
    print(num_paths(d, 1, 6))

# CONTEST IMPLEMENTATION
# general rule: bfs better bc of python recursion stack limit 
# need sys.setrecursionlimit() but that can cause MLE
'''
# kahn's algorithm
def top_sort_bfs(graph):
    ans = []
    indegree = {node: 0 for node in graph.keys()}
    for nodes in graph.values():
        for node in nodes:
            indegree[node] += 1
    q = deque( node for node in indegree if indegree[node] == 0 )
    while q:
        node = q.popleft()
        ans.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                q.append(neighbor)
    return [] if len(ans) != len(graph) else ans


def top_sort_dfs(graph):
    def dfs(node):
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]: dfs(neighbor)
        ans.append(node)
    visited = {node: False for node in graph.keys()}
    ans = []
    for node in graph.keys():
        if not visited[node]: dfs(node)
    ans = ans[::-1]
    pos = {node: idx for idx, node in enumerate(ans)}
    bad = any(pos[node] > pos[neighbor] for node in graph.keys() for neighbor in graph[node])
    return [] if bad else ans
'''