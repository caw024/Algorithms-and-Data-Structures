# from collections import defaultdict, Counter, deque
# from heapq import heappop, heappush, heapify
# from functools import lru_cache, reduce
# import bisect
# from itertools import permutations as p, combinations as c, combinations_with_replacement as cwr
# from math import factorial as f, sqrt, inf, gcd

import os, sys
def inp(): return sys.stdin.buffer.readline().rstrip()
def inpa(): return list(map(int, inp().split()))
def out(var): sys.stdout.write(str(var)+"\n")
def outa(var): sys.stdout.write(' '.join(map(str, var))+'\n')
# remove buffer when dealing w strings (they are in bytes, or convert them to strings)
# can also try:
# import io
# input = io.BytesIO(os.read(0,os.fstat(0).st_size)).readline


def redirect_io():
    import os, sys
    dir_path = os.path.dirname(os.path.realpath(__file__))
    sys.stdin = open(os.path.join(dir_path, "input.txt"), 'r')
    # sys.stdout = open(os.path.join(dir_path, "output.txt"), 'r')
redirect_io()


def solve(arr):
    ans = 0
    return ans


for _ in range(int(inp())):
    arr = inpa()
    ans = solve(arr)
    out(ans)


