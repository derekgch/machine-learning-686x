import numpy as np


def sigmoid(num):
  if num >= 1:
    return 1
  elif num <= -1:
    return 0
  else:
    return 1/(1 + np.exp(-num)) 
    
def tanh_fn(num):
  if num >= 1:
    return 1
  elif num <= -1:
    return -1
  else:
    return np.sinh(num) / np.cosh(num)



def lstm_hw(h_t_1, x_t, c_t_1):
  f_t=sigmoid(0+0-100)
  i_t=sigmoid(0+100*x_t+100)
  o_t=sigmoid(0+100*x_t+0)
  cell_tanh = (-100*h_t_1 + 50*x_t+0)
  c_t = f_t*c_t_1 + i_t*tanh_fn(cell_tanh)
  h_t = o_t*tanh_fn(c_t)
  if np.absolute(h_t) == 0.5:
    h_t=0
  # print('h_t_1, c_t_1, h_t', h_t_1, c_t_1, h_t)
  return c_t_1, h_t
  



def main(arr_1):
  h_1 = 0
  c_1 = 0
  result = []
  for n in arr_1:
    # print('n is ', n)
    c_t_1, h_t_1= lstm_hw(h_1,n,c_1 )

    c_1 = c_t_1
    h_1 = h_t_1
    result.append(h_1)
    
  print('seq is ', arr_1)
  print('result is ', result)

seq1 = [0,0,1,1,1,0]
seq2 = [1,1,0,1,1]
seq3 = [1,1,1,1,1]
seq4 = [0,1,1,1,1]
seq5 = [0,0,0,0,0]

main(seq1)
main(seq2)
main(seq3)
main(seq4)
main(seq5)
main([1,1,1,1,1,1,1,1,1,1,0])


    
