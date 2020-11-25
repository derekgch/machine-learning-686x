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
  # kmixture, kpost, kcost = kmeans.run(X, mixture, post)
  # title = f"K is {K}, seed is {seed}, cost is {kcost}"
  em_mixture, em_post, em_cost = naive_em.run(X, mixture, post)
  with_bic = common.bic(X, em_mixture, em_cost)
  title = f"K is {K}, seed is {seed}, em_cost is {em_cost}, with_bic is {with_bic}"
  print(title)
  common.plot(X, em_mixture, em_post, title)

# TODO: Your code here
