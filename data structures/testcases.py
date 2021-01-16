from random import randint


def generate_array(length, lower, upper):
    return [randint(lower, upper) for _ in range(length)]


def generate_test_cases(cases):
    N = 1000
    return [generate_array(randint(1, 1000), -N, N) for _ in range(cases)]


if __name__ == "__main__":
    testcases = generate_test_cases(10)
    for arr in testcases:
        print(arr)
