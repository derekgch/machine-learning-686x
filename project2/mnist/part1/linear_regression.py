import numpy as np

### Functions for you to fill in ###


def closed_form(X, Y, lambda_factor):
    """
    Computes the closed form solution of linear regression with L2 regularization

    Args:
        X - (n, d + 1) NumPy array (n datapoints each with d features plus the bias feature in the first dimension)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        lambda_factor - the regularization constant (scalar)
    Returns:
        theta - (d + 1, ) NumPy array containing the weights of linear regression. Note that theta[0]
        represents the y-axis intercept of the model and therefore X[0] = 1

    closed form solution:
    theta = (X^T * X + lambda*IdentityMatrix)^-1 * X^T*Y
    """
    # YOUR CODE HERE
    print(X, Y, lambda_factor)
    length_of_X = len(X[0])
    x_transpose = np.transpose(X)
    # print('x_transpose \n', x_transpose)

    x_trans_x_product = np.dot(x_transpose, X)
    # print('x_trans_x_product \n', x_trans_x_product)

    x_trans_y_product = np.dot(x_transpose, Y)
    # print('x_trans_y_product \n', x_trans_y_product)

    lambda_I_product = lambda_factor * np.identity(length_of_X)
    # print('lambda_I_product\n ', lambda_I_product)

    term_to_inverse = x_trans_x_product + lambda_I_product
    # print('term_to_inverse\n ', term_to_inverse)

    try:
        inverse = np.linalg.inv(term_to_inverse)

    except np.linalg.LinAlgError:
        # Not invertible. Skip this one.
        pass
    else:
        result = np.dot(inverse, x_trans_y_product)

        # print('inverse', inverse)
        # print('result', result)
        return result
        # raise NotImplementedError

        ### Functions which are already complete, for you to use ###


def compute_test_error_linear(test_x, Y, theta):
    test_y_predict = np.round(np.dot(test_x, theta))
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
