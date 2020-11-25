import numpy as np
import em
import common

# X = np.loadtxt("test_incomplete.txt")
X = np.loadtxt("netflix_incomplete.txt")


X_gold = np.loadtxt("netflix_complete.txt")

K = 12
n, d = X.shape
# seeds = [0,1,2,3,4]
seeds=[1]
for seed in seeds:
  mixture, post = common.init(X, K, seed)
  # kmixture, kpost, kcost = kmeans.run(X, mixture, post)
  # title = f"K is {K}, seed is {seed}, cost is {kcost}"
  em_mixture, em_post, em_cost = em.run(X, mixture, post)
  X_pred = em.fill_matrix(X, em_mixture)
  rmse = common.rmse(X_gold,X_pred)
  print(f'RMSE is {rmse}')
  # with_bic = common.bic(X, em_mixture, em_cost)
  title = f"K is {K}, seed is {seed}, em_cost is {em_cost}"
  print(title)
  # common.plot(X, em_mixture, em_post, title)
# TODO: Your code here
