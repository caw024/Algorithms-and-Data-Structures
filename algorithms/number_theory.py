from math import sqrt, gcd
from collections import defaultdict

def lcm(a,b):
    return abs(a*b) // gcd(a,b)

# O(sqrt(n)): prints divisors in sorted order
def divisors(n: int) -> list:
    divisors = []
    i = 1
    while i * i < n:
        if n % i == 0:
            divisors.append(i)
        i += 1
    if i * i == n:
        divisors.append(i)
    prev_i = i-1
    for i in range(prev_i, 0, -1):
        if n % i == 0:
            divisors.append(n//i)
    return divisors


# O(sqrt(n)): factors a number to primes, returns a dict
def factor(n: int) -> dict:
    d = defaultdict(int)
    while n % 2 == 0:
        d[2] += 1
        n = n//2

    for i in range(3, int(sqrt(n))+1, 2):
        while n % i == 0:
            d[i] += 1
            n = n//i
    # remaining number is a prime greater than sqrt(n)
    if n > 2:
        d[n] = 1
    return d


# O(sqrt(n))
def is_prime(n: int) -> bool:
    # if n is factorable, it has a prime less than or equal to sqrt(n)
    if n % 2 == 0:
        return False
    for i in range(3, int(sqrt(n))+1, 2):
        if n % i == 0:
            return False
    return True



# O(nloglog(n)) preprocessing
# O(logn) per QUERY 
def sieves(n: int) -> None:
    n += 1
    # stores smallest prime factor at each index i
    spf = [i for i in range(n+1)]
    for i in range(2, n+1):
        # smallest prime factor already found
        if spf[i] != i:
            continue
        # loop through when number is prime
        # every number under i^2 has smallest prime factor less than i
        for j in range(i**2, n+1, i):
            if spf[j] == j:
                spf[j] = i
    return spf
    
# example queries: prime factorization, is prime, etc.
spf = sieves(10**2)

# O(logn) w sieves
def factor2(n: int) -> dict:
    d = defaultdict(int)
    while n != 1:
        d[spf[n]] += 1
        n = n//spf[n]
    return d

# O(1) w sieves
def is_prime2(n: int) -> bool:
    return spf[n] == n

# O(nlogn) w seives
def gen_primes(n: int) -> list:
    spf = sieves(n + 1)
    return [i for i in range(n) if spf[i] == i]


def numFactors(n: int) -> int:
    # get factorization: can be O(sqrt(n)) OR O(logn) based on which factors
    d = factor(n)
    ans = 1
    for val in d.items():
        ans *= (val + 1)
    return ans
    

if __name__ == "__main__":
    print(spf)
    '''
    for i in range(1, 10**2):
        f= {k: v for k, v in factor(i).items()}
        print(f"{i}\tdivisors: {divisors(i)}")
        print(f"\tfactors:{f} \tis prime:{is_prime(i)}")
    '''


# CONTEST IMPLEMENTATION SIEVE
'''
spf = [i for i in range(n)]
for i in range(2, int(sqrt(n)) + 1):
    if spf[i] != i:
        continue
    for j in range(i**2, n, i):
        if spf[j] == j:
            spf[j] = i
'''