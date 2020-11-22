import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")
K =4
seeds = [0,1,2,3,4]
for seed in seeds:
  mixture, post = common.init(X, K, seed)
  kmixture, kpost, kcost = kmeans.run(X, mixture, post)
  title = f"K is {K}, seed is {seed}, cost is {kcost}"
  print(title)
  common.plot(X, kmixture, kpost, title)

# TODO: Your code here
