#######################################
import os
import sys
from io import BytesIO, IOBase

BUFSIZE = 8192

class FastIO(IOBase):
	newlines = 0

	def __init__(self, file):
		self._fd = file.fileno()
		self.buffer = BytesIO()
		self.writable = "x" in file.mode or "r" not in file.mode
		self.write = self.buffer.write if self.writable else None

	def read(self):
		while True:
			b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
			if not b:
				break
			ptr = self.buffer.tell()
			self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
		self.newlines = 0
		return self.buffer.read()

	def readline(self):
		while self.newlines == 0:
			b = os.read(self._fd, max(os.fstat(self._fd).st_size, BUFSIZE))
			self.newlines = b.count(b"\n") + (not b)
			ptr = self.buffer.tell()
			self.buffer.seek(0, 2), self.buffer.write(b), self.buffer.seek(ptr)
		self.newlines -= 1
		return self.buffer.readline()

	def flush(self):
		if self.writable:
			os.write(self._fd, self.buffer.getvalue())
			self.buffer.truncate(0), self.buffer.seek(0)


class IOWrapper(IOBase):
	def __init__(self, file):
		self.buffer = FastIO(file)
		self.flush = self.buffer.flush
		self.writable = self.buffer.writable
		self.write = lambda s: self.buffer.write(s.encode("ascii"))
		self.read = lambda: self.buffer.read().decode("ascii")
		self.readline = lambda: self.buffer.readline().decode("ascii")


sys.stdin, sys.stdout = IOWrapper(sys.stdin), IOWrapper(sys.stdout)
input = lambda: sys.stdin.readline().rstrip("\r\n")
#######################################

# import random
# RANDOM = random.randrange(2**62)

# def Wrapper(x):
#   return x ^ RANDOM

# to use: wx = Wrapper(x) as key in dict

#######################################


def redirect_io():
    import pathlib, sys
    fname = pathlib.Path(__file__).parent/"input.txt"
    sys.stdin = open(fname, 'r')
redirect_io()

def ina(): return list(map(int, input().split()))

'''
'''
def solve(n, arr):
    ans = 0
    return ans


ans = []
for _ in range(int(input())):
    n = int(input())
    arr = ina()
    ans.append(str(solve(n, arr)))
print('\n'.join(ans))