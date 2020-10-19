import numpy as np
from sklearn.svm import LinearSVC


### Functions for you to fill in ###

def one_vs_rest_svm(train_x, train_y, test_x, C=0.1):
    """
    Trains a linear SVM for binary classifciation

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (0 or 1) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (0 or 1) for each test data point
    """
    # print("C is ", C)
    linear_svc_ = LinearSVC(random_state=0, C=C)
    linear_svc_.fit(train_x, train_y)
    result = linear_svc_.predict(test_x)

    # print('result', result)
    return result
    # raise NotImplementedError


def multi_class_svm(train_x, train_y, test_x):
    """
    Trains a linear SVM for multiclass classifciation using a one-vs-rest strategy

    Args:
        train_x - (n, d) NumPy array (n datapoints each with d features)
        train_y - (n, ) NumPy array containing the labels (int) for each training data point
        test_x - (m, d) NumPy array (m datapoints each with d features)
    Returns:
        pred_test_y - (m,) NumPy array containing the labels (int) for each test data point
    """
    # raise NotImplementedError
    linear_svc_ = LinearSVC(random_state=0, C=0.1)
    linear_svc_.fit(train_x, train_y)
    result = linear_svc_.predict(test_x)

    # print('result', result)
    return result


def compute_test_error_svm(test_y, pred_test_y):
    return 1 - np.mean(pred_test_y == test_y)
