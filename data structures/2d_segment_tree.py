from segment_tree import SegmentTree

# Not complete

class TwoDimSegmentTree:
    def __init__(self, max_x, max_y):
        # need to add everything
        self.tree = [ [0]*(max_y + 1) for _ in range(max_x + 1) ]

    # def __repr__(self):
        # return ' '.join(list(map(str, self.tree)))

    # O(logn), computes sum of range query in arr[l:r+1]
    def rangeQuery(self, x, y):
        ans = 0
        while (x > 0):
            y_ = y
            while (y_ > 0):
                ans += self.tree[x][y_]
                y_ -= y_ & (y_ + 1)
            x -= x & (x + 1)
        return ans

    def get(self, x1, y1, x2, y2):
        if not(x1 <= x2 and y1 <= y2):
            return False
        return self.rangeQuery(x2, y2) - self.rangeQuery(x2, y1) - self.rangeQuery(x1, y2) + self.rangeQuery(x1, y1)

    # O(logn), updates index i of array with new_val
    def updatePoint(self, x, y, v):
        max_x, max_y = len(self.tree), len(self.tree[0])
        while x < max_x:
            print(f'{x=} {max_x=}')
            y_ = y
            while (y_ < max_y):
                print(f'{y=} {max_y=}')
                self.tree[x][y_] += v
                y_ += y_ & (y_ + 1)
            x += x & (x + 1)


if __name__ == "__main__":
    # from testcases import generate_test_cases
    from random import randint
    from collections import defaultdict

    max_x = max_y = 100
    st = TwoDimSegmentTree(max_x, max_y)
    N = 100
    d = defaultdict(int)

    # add points
    for i in range(N):
        x, y = randint(0, max_x), randint(0, max_y)
        v = randint(1, 10)
        d[(x, y)] = v
        st.updatePoint(x, y, v)

    # compute ranges
    for i in range(N):
        x, y = randint(0, max_x), randint(0, max_y)
        segtree_ans = st.rangeQuery(x, y)
        actual_ans = sum(d[(a,b)] for a, b in d.keys() if a <=x and b <= y)
        if (segtree_ans != actual_ans):
            print(f"False {x=} {y=} {segtree_ans=} {actual_ans=}")

    # # update some more points
    # for i in range(N):
    #     x, y = randint(0, max_x), randint(0, max_y)
    #     items.append((x, y))
    #     st.updatePoint(x, y, 1)

    # # compute more ranges
    # for i in range(N):
    #     x, y = randint(0, max_x), randint(0, max_y)
    #     segtree_ans = st.rangeQuery(x, y)
    #     actual_ans = sum(1 for a,b in items if a <=x and b <= y)
    #     if (segtree_ans != actual_ans):
    #         print(f"False {x=} {y=} {segtree_ans=} {actual_ans=}")

    print("Finished run")