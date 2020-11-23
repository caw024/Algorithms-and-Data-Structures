class DSU:
    def __init__(self, N):
        # stores parent node of given vertex
        self.parents = list(range(N))
        # gives size of component w given node
        self.size = [1] * N

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
        xr, yr = self.find(x), self.find(y)
        # if they're in same group, do nothing
        if xr == yr:
            return False

        if self.size[xr] > self.size[yr]:
            xr, yr = yr, xr
        # size of xr >= size of yr
        # smaller points to the larger
        self.parents[xr] = yr
        self.size[xr] += self.size[yr]
        self.size[yr] = self.size[xr]
        return True

    def isConnected(self,x,y):
        return self.find(x) == self.find(y)

    def getSize(self, x):
        return self.size[self.find(x)]

# APPLICATIONS
'''
Best for checking for connected components in graph
Or whether there exists a path from a to b
'''
if __name__ == "__main__":
    N = 7
    dsu = DSU(N)
    dsu.union(0,1)
    dsu.union(0,2)
    dsu.union(0,3)
    dsu.union(4,5)

    # {0} <- {1,2,3} and {4,5}
    print(f"Not-totally-updated size of connected components: {dsu.size}")
    for i in range(N): 
        print(f"Parent node of {i} is {dsu.find(i)}", f"\tSize of {i}th node is {dsu.getSize(i)}")
        # print(f"Is {i} connected to {(i+1)%N}: {dsu.isConnected(i,(i+1)%N)}")
        # print(f"Size of {i}th node is {dsu.getSize(i)}")
