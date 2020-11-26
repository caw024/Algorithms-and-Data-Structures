from collections import defaultdict

'''
Returns topologically sort list of nodes (if there is a cycle, return []),
Done with DFS
Runtime O(V+E), space O(V)
'''
def topological(d):
    # ASSUMES OUR NODES ARE LABELLED 0 TO N-1
    numv = len(d)
    process = [0]*numv
    ans = []

    # STATES OF NODES:
    # NOT_PROCESSED = 0
    UNDER_PROCESSING = 1
    PROCESSED = 2

    # scope lets us access process and answer easily
    def dfs(node):
        if process[node] is PROCESSED:
            return True
        # this gives us a cycle
        elif process[node] is UNDER_PROCESSING:
            return False
        
        # node has not been processed
        process[node] = UNDER_PROCESSING
        
        for neighbor in d[node]:
            if dfs(neighbor) == False:
                return False

        # add node in order of furthest reach
        ans.append(node)
        process[node] = PROCESSED
        return True

    if any(not dfs(i) for i in range(numv)):
        return []

    # gives the topologically sorted list of vertices
    return ans[::-1]

# APPLICATIONS:
'''
help determine a valid order to perform activities
can check for cycles in directed acyclic graph
'''
if __name__ == "__main__":
    d = defaultdict(list)
    d[0] = [1,4]
    d[1] = [2,3]
    d[2] = [3]
    d[3] = [4]
    d[4] = []
    print(topological(d))

    e = defaultdict(list)
    e[0] = [1,2]
    e[1] = [0,2]
    e[2] = []
    print(topological(e))

