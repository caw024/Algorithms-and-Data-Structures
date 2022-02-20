def inpa(): return list(map(int, input().split()))
def outa(var): print(' '.join(map(str, var)))

def redirect_io():
    import pathlib, sys
    fname = pathlib.Path(__file__).parent/"input.txt"
    sys.stdin = open(fname, 'r')
redirect_io()


def solve(arr):
    ans = 0
    return ans


for _ in range(int(input())):
    arr = inpa()
    ans = solve(arr)
    print(ans)


