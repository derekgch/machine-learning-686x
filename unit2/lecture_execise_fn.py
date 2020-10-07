import numpy as np


def compute_loss(theta, label, vector_x):
    result = label - np.dot(theta, vector_x)
    return result


def main():
    vectors = [[1, 0, 1], [1, 1, 1], [1, 1, -1], [-1, 1, 1]]
    labels = [2, 2.7, -0.7, 2]
    theta = [0, 1, 2]
    t = 4
    index = 0
    sum = 0
    for vector in vectors:
        result_z = compute_loss(theta, labels[index], vector)
        print('for index ', index, ', z is ', result_z)
        sum += (result_z**2)/2
        # if result_z < 1:
        # sum += (1-result_z)

        index += 1

    print('sum is', sum)
    print('risk is ', sum/t)


main()
