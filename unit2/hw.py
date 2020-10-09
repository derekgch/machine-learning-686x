import numpy as np

u = [6, 0, 3, 6]
v = [4, 2, 1]

print(u, v)

print(np.transpose(v))

print(np.outer(u, np.transpose(v)))
