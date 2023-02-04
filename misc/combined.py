# flows.py

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


# geometry.py

from math import sqrt, cos, sin, tan, acos, pi
from functools import partial

class Point:
    """ Every 2D point can be represented as tuple (x,y) """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.coor = (x,y)

    def __repr__(self):
        x, y = map(partial(round, ndigits=5), self.coor)
        return f"(x,y): ({x},{y})\t"

    def __eq__(self, other):
        return (abs(self.x - other.x) < 1e-8 and abs(self.y - other.y) < 1e-8)

    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Point(scalar * self.x, scalar * self.y)

    # for clockwise rotation, use negative angles
    def rotate_ccw_origin(self, angle):
        x2 = cos(angle) * self.x - sin(angle) * self.y
        y2 = sin(angle) * self.x + cos(angle) * self.y
        return Point(x2, y2)

    def rotate_ccw_90(self):
        return Point(-self.y, self.x)

    def rotate_ccw_point(self, angle, center_rotation):
        translated = (self - center_rotation)
        # print("translated", translated)
        rotated_pt = translated.rotate_ccw_origin(angle) + center_rotation
        # print("rotated pt", rotated_pt)
        return rotated_pt

    @staticmethod
    def dist(A, B):
        return sqrt((A.x - B.x)**2 + (A.y - B.y)**2)



class Line:
    def __init__(self, p1: Point = None, p2: Point = None, eqn: tuple = None):
        """Equation of a 2D line is of the form:
        Ax + By + C = 0
        """
        if p1 and p2:
            self.eqn = self._get_line(p1, p2)
        elif eqn:
            self.eqn = eqn
        else:
            raise Exception("Not a valid line")

    def __repr__(self):
        a, b, c = map(partial(round, ndigits=5), self.eqn)
        return f"Eqn: {a}x + {b}y + {c}\t"

    def _get_line(self, A, B) -> tuple:
        """ returns line of the form ax + by + c = 0 as (a,b,c)
        when c == 0, line is vertical. else c == 1 is nonvertical
        """
        Ax, Ay = A.coor
        Bx, By = B.coor
        # vertical line
        if Ax == Bx:
            return (-1, 0, Ax)
        M = (By - Ay)/(Bx - Ax)
        b = Ay - M * Ax
        return (M, -1, b)

    def eval_x(self, x) -> float:
        # solve for y given x
        a, b, c = self.eqn
        if b == 0:
            return None
        return (a * x + c) / -b

    def eval_y(self, y) -> float:
        # solve for x given y
        a, b, c = self.eqn
        if a == 0:
            return None
        return (b * y + c) / -a


def get_angle(a: Point, b: Point, c: Point) -> float:
    # get angle ABC in radians
    ab, bc, ac = Point.dist(a,b), Point.dist(b,c), Point.dist(a,c)
    return acos(((bc*bc + ab*ab)-ac*ac)/(2*bc*ab))



class Triangle:
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.AB = Line(A,B)
        self.BC = Line(B,C)
        self.AC = Line(A,C)
        # ab, bc, ac
        self.distances = [Point.dist(A,B), Point.dist(B,C), Point.dist(A,C)]
        # ABC, CAB, BCA (not directed - at most 180)
        self.angles = [get_angle(A,B,C), get_angle(C,A,B), get_angle(B,C,A)]

    def __repr__(self):
        A, B, C = self.A, self.B, self.C
        distances = self.distances
        angles = list(map(lambda x: x * 180 / pi, self.angles))
        s = []
        s.append(f"Points {A=} {B=} {C=}")
        s.append(f"{distances=}")
        s.append(f"In degrees: {angles=}")
        return '\n'.join(s)



# checks if line is parallel -> call this before line_intersect
def is_parallel(l1, l2) -> bool:
    a1, b1, _ = l1.eqn
    a2, b2, _ = l2.eqn
    if a1 == a2 == 0:
        return True
    if a1 == 0 or a2 == 0:
        return False
    return (b1/a1 == b2/a2)


# assumes lines aren't parallel; returns a point
def line_intersect(l1, l2) -> Point:
    a1, b1, c1 = l1.eqn
    a2, b2, c2 = l2.eqn

    x = (b1 * c2 - b2 * c1)/(a1 * b2 - a2 * b1)
    y = (a2 * c1 - a1 * c2)/(a1 * b2 - a2 * b1)
    return Point(x,y)


def perp_bis(A, B) -> Line:
    xmid = (A+B).x/2
    ymid = (A+B).y/2
    line = Line(A,B)
    M, c, _ = line.eqn
    # current line is horizontal, perp bis has constant x
    if M == 0:
        return Line(Point(xmid, 0), Point(xmid, 1))
    # current line is vertical line, perp bis has constant y
    if c == 0:
        return Line(Point(0, ymid), Point(1, ymid))
    M2 = -1/M
    b2 = ymid - xmid * M2
    return Line(eqn=(M2, -1, b2))


def reflect(point, line) -> Point:
    proj = projection(point, line)
    return 2 * proj - point


def projection(point, line) -> Point:
    x1, y1 = point.coor
    M, c, b = line.eqn
    if M == 0:
        return Point(x1, b)
    if c == 0:
        return Point(b, y1)
    M2 = -1/M
    b2 = y1 - M2 * x1
    return line_intersect(line, Line(eqn=(M2, -1, b2)) )


def heron(A, B, C) -> float:
    a, b, c = Point.dist(B,C), Point.dist(A, C), Point.dist(A, B)
    s = (a+b+c)/2
    return sqrt(s * (s-a) * (s-b) * (s-c))


# def distance_from_point_to_line(point, line):
#     return dist(point, projection(point, line))

def distance_from_point_to_line(point, line) -> float:
    a, b, c = line.eqn
    x,y = point.coor
    if abs(a) + abs(b) == 0:
        return 0
    return abs(a*x + b*y + c)/sqrt(a**2 + b**2)


def rotated_line_around_point(rotated_center, line, angle) -> Line:
    # ccw rotation of line at some angle (up to 90 degrees) wrt rotated_center (on line)
    # get another point on the line
    p0 = Point(0, line.eval_x(0)) if line.eval_x(0) is not None else Point(line.eval_y(0), 0)
    p1 = Point(1, line.eval_x(1)) if line.eval_x(1) is not None else Point(line.eval_y(1), 1)
    perp_foot = p0 if rotated_center != p0 else p1

    triangle_base = perp_foot - rotated_center
    triangle_height = triangle_base.rotate_ccw_90() * tan(angle)
    triangle_pt = perp_foot + triangle_height
    return Line(rotated_center, triangle_pt)


if __name__ == "__main__":
    # methods have been tested on Three Triangle problem of 2020 ICPC GRNY
    pass

# graphs.py

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
def dfs(node, visited):
    if node in visited:
        return
    visited.add(node)
    for neighbor in graph[node]:
        dfs(neighbor, visited)

# O(V+E)
def bfs(node):
    visited = {node}
    q = deque([node])
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


# numbertheory.py

from math import sqrt, gcd
from collections import defaultdict

def lcm(a,b):
    return abs(a*b) // gcd(a,b)

# O(sqrt(n)): prints divisors in sorted order
def divisors(n: int) -> list:
    divisors = []
    i = 1
    while i * i < n:
        if n % i == 0:
            divisors.append(i)
        i += 1
    if i * i == n:
        divisors.append(i)
    prev_i = i-1
    for i in range(prev_i, 0, -1):
        if n % i == 0:
            divisors.append(n//i)
    return divisors


# O(sqrt(n)): factors a number to primes, returns a dict
def factorize(n: int) -> dict:
    d = defaultdict(int)
    while n % 2 == 0:
        d[2] += 1
        n = n//2

    for i in range(3, int(sqrt(n))+1, 2):
        while n % i == 0:
            d[i] += 1
            n = n//i
    # remaining number is a prime greater than sqrt(n)
    if n > 2:
        d[n] = 1
    return d


# O(sqrt(n))
def is_prime(n: int) -> bool:
    # if n is factorable, it has a prime less than or equal to sqrt(n)
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(sqrt(n))+1, 2):
        if n % i == 0:
            return False
    return True



# O(nloglog(n)) preprocessing
# O(logn) per QUERY
def sieves(n: int) -> None:
    n += 1
    # stores smallest prime factor at each index i
    spf = [i for i in range(n+1)]
    for i in range(2, n+1):
        # smallest prime factor already found
        if spf[i] != i:
            continue
        # loop through when number is prime
        # every number under i^2 has smallest prime factor less than i
        for j in range(i**2, n+1, i):
            if spf[j] == j:
                spf[j] = i
    return spf

# example queries: prime factorization, is prime, etc.
spf = sieves(10**2)

# O(logn) w sieves
def factor2(n: int) -> dict:
    d = defaultdict(int)
    while n != 1:
        d[spf[n]] += 1
        n = n//spf[n]
    return d

def num_factors(n):
    d = factor2(n)
    ans = 1
    for i in d.values():
        ans *= (i+1)
    return ans

# O(1) w sieves
def is_prime2(n: int) -> bool:
    return spf[n] == n

# O(nlogn) w seives
def gen_primes(n: int) -> list:
    spf = sieves(n + 1)
    return [i for i in range(n) if spf[i] == i]


def num_factors(n: int) -> int:
    # get factorization: can be O(sqrt(n)) OR O(logn) based on which factors
    d = factorize(n)
    ans = 1
    for val in d.items():
        ans *= (val + 1)
    return ans

# O(sqrtn)
def phi(n):
    result = n
    i = 2
    while (i * i <= n):
        if (n % i == 0):
            while (n % i == 0):
                n //= i
            result -= result // i
        i += 1

    if (n > 1):
        result -= result // n
    return result

def inv(n, mod):
    return pow(n, mod-2, mod)



def factorials(MAXN, mod):
    ''' Precomputes list of MAXN factorials and
    inverse factorials of certain mod (O(MAXN))'''
    # MAXN = 3 * 10**5
    # mod = 998244353

    # pow(x, e, m) uses binary exponentiation
    # have to precompute factorial and inverse factorial because it can be VERY large and slow

    # fac has array [0, ..., MAXN]
    # invf[i] = (fac[i] ^ (mod-2)) % mod = (i+1) * invf[i+1] % mod

    fac = [1]
    for i in range(1, MAXN + 1):
        fac.append((fac[-1] * i) % mod)

    invf = [pow(fac[-1], mod-2, mod)]
    for i in range(MAXN, 0, -1):
        invf.append( (i * invf[-1]) % mod )
    invf = invf[::-1]
    return fac, invf

fac, invf = factorials(MAXN=1e5, mod=1e9+7)
def c(n, k, mod):
    return (fac[n] * invf[n-k] * invf[k]) % mod

if __name__ == "__main__":
    print(spf)
    '''
    for i in range(1, 10**2):
        f= {k: v for k, v in factor(i).items()}
        print(f"{i}\tdivisors: {divisors(i)}")
        print(f"\tfactors:{f} \tis prime:{is_prime(i)}")
    '''


# CONTEST IMPLEMENTATION SIEVE
'''
spf = [i for i in range(n)]
for i in range(2, int(sqrt(n)) + 1):
    if spf[i] != i:
        continue
    for j in range(i**2, n, i):
        if spf[j] == j:
            spf[j] = i
'''


# topological.py

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


# tree_algos.py

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


# dsu.py

from collections import defaultdict

class DSU:
    def __init__(self, N):
        # stores parent node of given vertex
        self.parents = list(range(N))
        # gives size of component w given node. Note: only accurate for root node
        self.size = [1] * N
        self.num_components = N
        # stores order of adding points
        self.stack = []

    # runtime: amoritized O(1)
    # returns root node
    def find(self, x):
        root = x
        # finds parent node at the end of the chain
        while self.parents[root] != root:
            root = self.parents[root]

        # optional: path compression
        while (x != root):
            nextx = self.parents[x]
            self.parents[x] = root
            x = nextx
        return root

    # runtime: amortized O(1) by combining union by size and path compression
    # unifies two components together (the ones containing x and y)
    def union(self, x, y):
        self.stack.append((x, y))
        xr, yr = self.find(x), self.find(y)
        # if they're in same group, do nothing
        if xr == yr:
            return False

        if self.size[xr] > self.size[yr]:
            xr, yr = yr, xr

        self.num_components -= 1

        # size of xr <= size of yr, so smaller points to the larger
        self.parents[xr] = yr
        self.size[yr] += self.size[xr]
        return True

    # do x and y share the same root node
    def isConnected(self, x, y):
        return self.find(x) == self.find(y)

    # get size of tree containing x
    def getSize(self, x):
        return self.size[self.find(x)]

    # get all connected components as a hash table, mapping root to set of nodes connected to root
    def getComponents(self):
        d = defaultdict(set)
        for i in range(len(self.parents)):
            d[self.find(i)].add(i)
        return d

    def undo(self):
        x, y = self.stack.pop()
        if (x == y):
            return
        self.parents[x] = x
        self.size[x] -= self.size[y]
        self.num_components += 1





# APPLICATIONS
'''
Best for checking for connected components in graph
Or whether there exists a path from a to b
'''
if __name__ == "__main__":
    N = 7
    dsu = DSU(N)
    dsu.union(0, 1)
    dsu.union(0, 2)
    dsu.union(0, 3)
    dsu.union(4, 5)

    # {0} <- {1,2,3} and {4,5}
    print(f"Not-totally-updated size of connected components: {dsu.size}")
    print(f"Connected Components given root: {dsu.getComponents()}")
    for i in range(N):
        print(f"Parent node of {i} is {dsu.find(i)}",
              f"\tSize of {i}th node is {dsu.getSize(i)}")
        # print(f"Is {i} connected to {(i+1)%N}: {dsu.isConnected(i,(i+1)%N)}")
        # print(f"Size of {i}th node is {dsu.getSize(i)}")



# Contest Implementation:
'''
class DSU:
    def __init__(self, N):
        self.p = list(range(N))
        self.sz = [1] * N

    def find(self, x):
        root = x
        while self.parents[root] != root:
            root = self.parents[root]
        while (x != root):
            nextx = self.parents[x]
            self.parents[x] = root
            x = nextx
        return root

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
    def numComponents(self):
        return len(set(self.find(x) for x in range(len(self.p))))
'''

# segment_tree.py

# Implementation 1: Using Array
class SegmentTree:
    # O(n), create an array of length 2n
    # child of i is arr[2*i] and arr[2*i+1]
    def __init__(self, arr):
        # note index 0 means nothing, works for n if power of 2
        n = len(arr)
        self.tree = [0]*n + arr
        # construct tree
        for i in range(n-1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def __repr__(self):
        return ' '.join(list(map(str, self.tree)))

    # O(logn), computes sum of range query in arr[l:r+1]
    def rangeQuery(self, l, r):
        ans = 0
        n = len(self.tree)//2
        l, r = l+n, r+n

        while l <= r:
            if (l & 1):
                ans += self.tree[l]
                l += 1
            if (r & 1 == 0):
                ans += self.tree[r]
                r -= 1
            l, r = l//2, r//2
        return ans

    # O(logn), updates index i of array with new_val
    def updatePoint(self, i, new_val):
        n = len(self.tree)//2
        self.tree[i + n] = new_val
        i = i + n

        # modify depending on the type of query
        while i >= 1:
            i = i//2
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]

        '''
        alternatively,
        while i > 1:
            self.tree[i//2] = self.tree[i] + self.tree[i ^ 1]
            i = i//2
        '''


# Implementation 2: Recursion
# not as optimized for python

from math import inf

class MaxSegmentTree:
    def __init__(self, arr):
        n = len(arr)
        self.tree = [0]*n + arr
        for i in range(n-1, 0, -1):
            # replace with appropriate query
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])

    def rangeQuery(self, l, r):
        ans = -inf
        n = len(self.tree)//2
        l, r = l+n, r+n
        while l <= r:
            if (l % 2):
                ans = max(ans, self.tree[l])
                l += 1
            if (r % 2 == 0):
                ans = max(ans, self.tree[r])
                r -= 1
            l, r = l//2, r//2
        return ans

    def updatePoint(self, i, new_val):
        n = len(self.tree)//2
        self.tree[i + n] = new_val
        i = i + n

        while i >= 1:
            i = i//2
            # replace with appropriate query
            self.tree[i] = max(self.tree[2 * i], self.tree[2 * i + 1])


if __name__ == "__main__":
    from testcases import generate_test_cases
    from random import randint

    numtests = 3
    testcases = generate_test_cases(numtests)
    for arr in testcases:
        st = MaxSegmentTree(arr)
        N = 100

        # print(arr)
        # print(st.tree)

        # compute ranges
        for i in range(N):
            l, r = sorted([randint(0, len(arr)-1), randint(0, len(arr)-1)])
            if max(arr[l:r+1]) != st.rangeQuery(l, r):
                print(f"False {l=} {r=} {max(arr[l:r+1])=} {st.rangeQuery(l, r)=}")

        # update some points
        for i in range(N):
            idx, val = randint(0, len(arr)-1), randint(-100, 100)
            st.updatePoint(idx, val)
            arr[idx] = val

        # compute ranges
        for i in range(N):
            l, r = sorted([randint(0, len(arr)-1), randint(0, len(arr)-1)])
            if max(arr[l:r+1]) != st.rangeQuery(l, r):
                print(f"False {l=} {r=} {max(arr[l:r+1])=} {st.rangeQuery(l, r)=}")

    print("Finished run")

    '''
    arr1 = [5, 8, 6, 3, 2, 7, 2, 6]
    arr2 = [3, 4, -2, 7, 3, 11, 5, -8, -9, 2, 4, -8]
    testcases = [arr1, arr2]

    for arr in testcases:
        print("initial array: ", arr)
        st = SegmentTree(arr)
        print("segment tree: ", st.tree)

        print("range queries:")
        print("[2,5]", st.rangeQuery(2, 5), arr[2:6], sum(arr[2:6]))
        print("[1,6]", st.rangeQuery(1, 6), arr[1:7], sum(arr[1:7]))
        print(f"[0,{len(arr)-1}]", st.rangeQuery(0, len(arr)-1), arr, sum(arr))

        print("\npoint updates:")
        st.updatePoint(5, 8)
        st.updatePoint(len(arr)-1, 10)
        arr[5] = 8
        arr[-1] = 10
        print("changed array:", arr)
        print("tree after updates: ", st.tree)

        print("range queries, again:")
        print("[2,5]", st.rangeQuery(2, 5), arr[2:6], sum(arr[2:6]))
        print("[1,6]", st.rangeQuery(1, 6), arr[1:7], sum(arr[1:7]))
        print(f"[0,{len(arr)-1}]", st.rangeQuery(0, len(arr)-1), arr, sum(arr))

        print('-'*20, '\n')
    '''

# Contest Implementation
'''
class SegmentTree:
    def __init__(self, arr):
        n = len(arr)
        self.tree = [0]*n + arr
        for i in range(n-1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def rangeQuery(self, l, r):
        ans = 0
        n = len(self.tree)//2
        l, r = l+n, r+n
        while l <= r:
            if (l & 1):
                ans += self.tree[l]
                l += 1
            if (r & 1 == 0):
                ans += self.tree[r]
                r -= 1
            l, r = l//2, r//2
        return ans

    def updatePoint(self, i, new_val):
        n = len(self.tree)//2
        self.tree[i + n] = new_val
        i = i + n
        while i >= 1:
            i = i//2
            self.tree[i] = self.tree[2*i] + self.tree[2*i+1]
'''

# sparse-table.py

from collections import defaultdict

# sparse table
# can do range queries i.e. min, max, lcm, gcd
#   -(not sum, which is not a range query)
# array must be static
sparse_table = defaultdict(int)


# get static min queries where r-l+1 is a power of 2, notice [l,r]
# O(nlogn) preprocessing to get min queries
def generate_sparse_table(arr):
    pow2 = 1
    # for each power of 2 (logn)
    while pow2 <= len(arr):
        r = pow2 - 1

        # get min values of [l,r] where r-l+1 = power of 2 (n)
        while r < len(arr):
            l = r - (pow2 - 1)

            if l == r:
                sparse_table[(l, r)] = arr[l]
                r += 1
                continue

            # O(1) computation, get power of 2 in between l and r
            w = (r - l + 1)//2

            # min can be replaced with max or gcd
            sparse_table[(l, r)] = min(sparse_table[(l, l + w - 1)],
                                       sparse_table[(l + w, r)]
                                       )
            r += 1

        pow2 = pow2 << 1


# O(1) access, min(arr[l,r]) including at r
def min_query(l, r):
    # get largest power of 2 no greater than r - l + 1
    k = 1
    while (k << 1) <= r - l + 1:
        k = k << 1

    return min(sparse_table[(l, l + k - 1)],
               sparse_table[(r - (k - 1), r)]
               )


if __name__ == "__main__":
    arr1 = [1, 3, 4, 8, 6, 1, 4, 2]
    generate_sparse_table(arr1)
    print(arr1)
    for l in range(len(arr1)):
        for r in range(l, len(arr1)):
            print(l, r, min_query(l, r))
    print('\n')

    arr2 = [7, 2, 3, 0, 5, 10, 3, 12, 18]
    generate_sparse_table(arr2)
    print(arr2)
    for l in range(len(arr2)):
        for r in range(l, len(arr2)):
            print(l, r, min_query(l, r))

# Contest Implementation
'''
from collections import defaultdict

sparse_table = defaultdict(int)

def generate_sparse_table(arr):
    pow2 = 1
    while pow2 <= len(arr):
        r = pow2 - 1
        while r < len(arr):
            l = r - (pow2 - 1)
            if l == r:
                sparse_table[(l, r)] = arr[l]
                r += 1
                continue
            w = (r - l + 1)//2
            sparse_table[(l, r)] = min(sparse_table[(l, l + w - 1)],
                                       sparse_table[(l + w, r)])
            r += 1
        pow2 = pow2 << 1

def min_query(l, r):
    k = 1
    while (k << 1) <= r - l + 1:
        k = k << 1
    return min(sparse_table[(l, l + k - 1)],
               sparse_table[(r - (k - 1), r)])
'''