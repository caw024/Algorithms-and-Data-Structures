# Kadane's Algorithm: O(N)
# Maximum Subarray Sum
def max_array_sum(arr):
    ans = prev_sum = 0
    for i in range(len(arr)):
        ans = max(ans, prev_sum + arr[i])
        prev_sum = max(prev_sum + arr[i], 0)
    return ans

print( max_array_sum([-1,2,4,-3,5,2,-5,2]) )