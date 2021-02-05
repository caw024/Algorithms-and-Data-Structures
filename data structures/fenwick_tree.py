class Fenwick:
    # O(n), constructs fenwick tree from given array
    def __init__(self, arr):
        self.arr = arr
        self.tree = [0] * (len(self.arr) + 1)

        for i in range(1, len(self.tree)):
            self.tree[i] = arr[i-1]

        # note index i in arr corresponds to i+1 in fenwick tree
        for i in range(1, len(self.tree)):
            parent_idx = i + self.LSB(i)
            if parent_idx >= len(self.tree):
                continue
            self.tree[parent_idx] += self.tree[i]

    # returns least significant bit, or right most bit
    def LSB(self, i):
        return i & (-i)

    # O(logn), prefix sum from arr[0:i+1] or tree[1:i+2] (inclusive)
    def prefixSum(self, i):
        prefix_sum = 0
        i = i+1
        while i > 0:
            prefix_sum += self.tree[i]
            i -= self.LSB(i)
        return prefix_sum

    # O(logn), computes sum of range query in arr[i:j+1]
    def rangeQuery(self, l, r):
        return self.prefixSum(r) - self.prefixSum(l-1)

    # O(logn), updates index idx of array with new_val
    def updatePoint(self, idx, new_val):
        dval = new_val - self.arr[idx]
        self.arr[idx] = new_val
        idx = idx+1

        while idx < len(self.tree):
            self.tree[idx] += dval
            idx += self.LSB(idx)


if __name__ == "__main__":
    arr = [3, 4, -2, 7, 3, 11, 5, -8, -9, 2, 4, -8]
    tree = [0, 3, 7, -2, 12, 3, 14, 5, 23, -9, -7, 4, -11]

    ft = Fenwick(arr)
    print("initial array: ", ft.arr)
    print("fenwick tree: ", ft.tree)
    '''
    print(ft.rangeQuery(2,5), arr[2:6])
    print(ft.rangeQuery(4,9), arr[4:10])
    print(ft.rangeQuery(0,len(arr)-1), arr)
    '''
    ft.updatePoint(1, 3)
    ft.updatePoint(len(arr)-1, 10)
    print("array after updates: ", ft.arr)
    '''
    print(ft.rangeQuery(2,5), arr[2:6])
    print(ft.rangeQuery(4,9), arr[4:10])
    print(ft.rangeQuery(0,len(arr)-1), arr)
    '''

# Contest Implementation
'''
class Fenwick:
    def __init__(self, arr):
        self.arr = arr
        self.tree = [0] * (len(self.arr) + 1)
        for i in range(1, len(self.tree)):
            self.tree[i] = arr[i-1]
        for i in range(1, len(self.tree)):
            parent_idx = i + self.LSB(i)
            if parent_idx >= len(self.tree):
                continue
            self.tree[parent_idx] += self.tree[i]

    def LSB(self, i):
        return i & (-i)

    def prefixSum(self, i):
        prefix_sum, i = 0, i+1
        while i > 0:
            prefix_sum, i = prefix_sum + self.tree[i], i - self.LSB(i)
        return prefix_sum

    def rangeQuery(self, l, r):
        return self.prefixSum(r) - self.prefixSum(l-1)

    def updatePoint(self, idx, new_val):
        dval = new_val - self.arr[idx]
        self.arr[idx] = new_val
        idx = idx+1
        while idx < len(self.tree):
            self.tree[idx] += dval
            idx += self.LSB(idx)
'''