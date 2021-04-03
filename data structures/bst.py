from collections import defaultdict

d = defaultdict(list)

class Node:
    def __init__(self, val, left=None, right=None, size=1):
        self.val = val
        self.left = left
        self.right = right
        self.size = size


# return 1 if d[a] > d[b] (d maps to value of a and b)
def comparator(a, b):
    return 1 if d[a] > d[b] or (a > b and d[a] == d[b]) else 0
    

class BST:
    def __init__(self, node=None, compare=comparator):
        self.root = node
        self.compare = compare

    # insert node into bst
    def insert(self, val):
        if self.root is None:   
            self.root = Node(val)
            return
        self._insert(self.root, val)

    # balance it!
    def _insert(self, node, val):
        node.size += 1
        # go left (smaller val)
        if self.compare(node.val, val):
            if node.left is None:
                node.left = Node(val)
            else:
                self._insert(node.left, val)
        # go right (larger or equal to val)
        else:
            if node.right is None:
                node.right = Node(val)
            else:
                self._insert(node.right, val)

    # returns size of removed val
    def remove(self, val):
        pass

    # checks if val is inBST
    def inBST(self, val):
        pass


bst = BST()