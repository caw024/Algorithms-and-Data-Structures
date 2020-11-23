from collections import defaultdict

# Detect if cycle in graph: Topological sort (if can't, there is cycle),
# Then pick the undirected edges that preserves the ordering
# returns list of topological sort
# in each dfs, traverse each vertex at most twice
# runtime O(V+E), space O(V)
d = defaultdict(int)

def topological(numv):
    process = [0]*numv
    ans = []

    # STATES: 0 = Not processed, 1 = Under Processing, 2 = Processed
    # scope lets us access process and ans easily
    def dfs(node):
        # if node has already been checked
        if process[node] == 2:
            return True
        # this gives us a cycle
        if process[node] == 1:
            return False
        process[node] = 1
        
        for neighbor in d[node]:
            if dfs(neighbor) == False:
                return False

        # will add node in order of furthest reach
        ans.append(node)
        process[node] = 2
        return True

    # ASSUMES OUR NODES ARE LABELLED 0 TO N-1
    for i in range(numv):
        if dfs(i) == False:
            return []

    # gives the topologically sorted list of vertices
    return ans[::-1]

# APPLICATIONS:
'''
help determine a valid order to perform activities
can check for cycles in directed acyclic graph
'''