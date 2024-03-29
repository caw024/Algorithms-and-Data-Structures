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
