from collections import defaultdict

# sparse table
# can do max and range gcd queries
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