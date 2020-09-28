import numpy as np


data_x = [[-1, -1], [1, 0], [-1, 10]]
label_y = [1, -1, 1]


def isError(theta, x, y):
    print('x and y', x, y)
    result = y*np.dot(theta, x)
    print('result', result)

    if result > 0:
        return False
    else:
        return True


def update_theta(theta, value, label):
    return theta + value*label


def loop_thru(pos, theta):
    count = 0
    while(count < 3):
        if isError(theta, np.array(data_x[pos]), label_y[pos]) is True:
            return pos
        else:
            count += 1
            pos += 1
            pos %= 3
    return False


def perceptron():
    theta = np.array([0, 0])
    result = loop_thru(1, theta)
    count = 0
    while(count < 20 and result is not False):
        theta = update_theta(theta, np.array(data_x[result]), label_y[result])
        print('theta is updated', theta, result+1, count)
        result = loop_thru((result+1) % 3, theta)
        count += 1


perceptron()
