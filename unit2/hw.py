import numpy as np

u = [6, 0, 3, 6]
v = [4, 2, 1]

print(u, v)

print(np.transpose(v))

print(np.outer(u, np.transpose(v)))


result = [5-24, 7-6, 2-0, 4-12, 3-12, 6-6]


def square_error(arr):
    result = 0
    for ele in arr:
        result += ele**2
    return result/2


print(result)
print(square_error(result))

print(square_error(u) + square_error(v))
