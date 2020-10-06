from string import punctuation, digits
import numpy as np
import random

# Part I


# pragma: coderesponse template
def get_order(n_samples):
    try:
        with open(str(n_samples) + '.txt') as fp:
            line = fp.readline()
            return list(map(int, line.split(',')))
    except FileNotFoundError:
        random.seed(1)
        indices = list(range(n_samples))
        random.shuffle(indices)
        return indices
# pragma: coderesponse end


# pragma: coderesponse template
def hinge_loss_single(feature_vector, label, theta, theta_0):
    """
    Finds the hinge loss on a single data point given specific classification
    parameters.

    Args:
        feature_vector - A numpy array describing the given data point.
        label - A real valued number, the correct classification of the data
            point.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given data point and parameters.
    loss fn = Loss_h(y^i*(theta dot x^i + theta_0)) 
    Z = y^i*(theta dot x^i + theta_0)
    loss_h = 0 if Z >=1, 1-Z if < 1
    """
    # Your code here
    result_z = label*(np.dot(theta, feature_vector) + theta_0)
    if result_z < 1:
        return 1-result_z
    else:
        return 0
# pragma: coderesponse end


# pragma: coderesponse template
def hinge_loss_full(feature_matrix, labels, theta, theta_0):
    """
    Finds the total hinge loss on a set of data given specific classification
    parameters.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.


    Returns: A real number representing the hinge loss associated with the
    given dataset and parameters. This number should be the average hinge
    loss across all of the points in the feature matrix.
    """
    # Your code here
    total_z = 0
    count = 0
    for feature_vector in feature_matrix:
        result_z = labels[count]*(np.dot(theta, feature_vector) + theta_0)
        if result_z < 1:
            total_z += 1-result_z
        count += 1

    return total_z/len(feature_matrix)

    # raise NotImplementedError
# pragma: coderesponse end


# pragma: coderesponse template
def perceptron_single_step_update(
        feature_vector,
        label,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the perceptron algorithm.

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        current_theta - The current theta being used by the perceptron
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the perceptron
            algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """
    # Your code here
    result_z = label * \
        (np.dot(current_theta, feature_vector) + current_theta_0)

    if result_z <= 0:
        new_theta = current_theta + label*feature_vector
        new_theta_0 = current_theta_0 + label
        return (new_theta, new_theta_0)
    else:
        return(current_theta, current_theta_0)

    # raise NotImplementedError
# pragma: coderesponse end


# pragma: coderesponse template
def perceptron(feature_matrix, labels, T):
    """
    Runs the full perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])

    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    theta, the linear classification parameter, after T iterations through the
    feature matrix and the second element is a real number with the value of
    theta_0, the offset classification parameter, after T iterations through
    the feature matrix.
    """
    theta = [0 for i in range(len(feature_matrix[0]))]
    theta_0 = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            result = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
            theta = result[0]
            theta_0 = result[1]
    return (theta, theta_0)
# pragma: coderesponse end


# pragma: coderesponse template
def average_perceptron(feature_matrix, labels, T):
    """
    Runs the average perceptron algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    NOTE: Iterate the data matrix by the orders returned by get_order(feature_matrix.shape[0])


    Args:
        feature_matrix -  A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the perceptron algorithm
            should iterate through the feature matrix.

    Returns: A tuple where the first element is a numpy array with the value of
    the average theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the average theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.

    Hint: It is difficult to keep a running average; however, it is simple to
    find a sum and divide.
    """
    # Your code here
    theta = [0 for i in range(len(feature_matrix[0]))]
    theta_0 = 0
    theta_total = [0 for i in range(len(feature_matrix[0]))]
    theta_0_total = 0
    total_update = 0
    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            result = perceptron_single_step_update(
                feature_matrix[i], labels[i], theta, theta_0)
            theta = result[0]
            theta_0 = result[1]
            total_update += 1
            theta_total = theta_total + theta
            theta_0_total = theta_0_total + theta_0

    return (theta_total/total_update, theta_0_total/total_update)
    # raise NotImplementedError
# pragma: coderesponse end


# pragma: coderesponse template
def pegasos_single_step_update(
        feature_vector,
        label,
        L,
        eta,
        current_theta,
        current_theta_0):
    """
    Properly updates the classification parameter, theta and theta_0, on a
    single step of the Pegasos algorithm

    Args:
        feature_vector - A numpy array describing a single data point.
        label - The correct classification of the feature vector.
        L - The lamba value being used to update the parameters.
        eta - Learning rate to update parameters.
        current_theta - The current theta being used by the Pegasos
            algorithm before this update.
        current_theta_0 - The current theta_0 being used by the
            Pegasos algorithm before this update.

    Returns: A tuple where the first element is a numpy array with the value of
    theta after the current update has completed and the second element is a
    real valued number with the value of theta_0 after the current updated has
    completed.
    """

    result_z = label*(np.dot(current_theta, feature_vector) + current_theta_0)
    if result_z <= 1:
        new_theta = np.dot((1-eta*L), current_theta) + \
            np.dot(eta*label, feature_vector)
        new_theta_0 = current_theta_0 + eta*label
        return new_theta, new_theta_0
    else:
        new_theta = (1-eta*L) * current_theta
        return new_theta, current_theta_0

    # Your code here
    # raise NotImplementedError
# pragma: coderesponse end


# pragma: coderesponse template
def pegasos(feature_matrix, labels, T, L):
    """
    Runs the Pegasos algorithm on a given set of data. Runs T
    iterations through the data set, there is no need to worry about
    stopping early.

    For each update, set learning rate = 1/sqrt(t),
    where t is a counter for the number of updates performed so far (between 1
    and nT inclusive).

    NOTE: Please use the previously implemented functions when applicable.
    Do not copy paste code from previous parts.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
        labels - A numpy array where the kth element of the array is the
            correct classification of the kth row of the feature matrix.
        T - An integer indicating how many times the algorithm
            should iterate through the feature matrix.
        L - The lamba value being used to update the Pegasos
            algorithm parameters.

    Returns: A tuple where the first element is a numpy array with the value of
    the theta, the linear classification parameter, found after T
    iterations through the feature matrix and the second element is a real
    number with the value of the theta_0, the offset classification
    parameter, found after T iterations through the feature matrix.
    """
    theta = [0 for i in range(len(feature_matrix[0]))]
    theta_0 = 0
    counter = 0
    # Your code here

    def single_update(feature_vector,
                      label,
                      L,
                      eta,
                      current_theta,
                      current_theta_0):
        result_z = label * \
            (np.dot(current_theta, feature_vector) + current_theta_0)
        if result_z <= 1:
            new_theta = np.dot((1-eta*L), current_theta) + \
                np.dot(eta*label, feature_vector)
            new_theta_0 = current_theta_0 + eta*label
            return new_theta, new_theta_0
        else:
            new_theta = (1-eta*L) * current_theta
            return new_theta, current_theta_0

    for t in range(T):
        for i in get_order(feature_matrix.shape[0]):
            counter += 1
            eta = 1 / counter ** 0.5
            result = single_update(
                feature_matrix[i], labels[i], L, eta, theta, theta_0)
            theta = result[0]
            theta_0 = result[1]

    return (theta, theta_0)

    # raise NotImplementedError
# pragma: coderesponse end

# Part II


# pragma: coderesponse template
def classify(feature_matrix, theta, theta_0):
    """
    A classification function that uses theta and theta_0 to classify a set of
    data points.

    Args:
        feature_matrix - A numpy matrix describing the given data. Each row
            represents a single data point.
                theta - A numpy array describing the linear classifier.
        theta - A numpy array describing the linear classifier.
        theta_0 - A real valued number representing the offset parameter.

    Returns: A numpy array of 1s and -1s where the kth element of the array is
    the predicted classification of the kth row of the feature matrix using the
    given theta and theta_0. If a prediction is GREATER THAN zero, it should
    be considered a positive classification.
    """
    # Your code here
    classifies = []
    for vector in feature_matrix:
        result = theta_0 + np.dot(vector, theta)
        if result > 0:
            classifies.append(1)
        else:
            classifies.append(-1)

    return np.array(classifies)
    # raise NotImplementedError
# pragma: coderesponse end


# pragma: coderesponse template
def classifier_accuracy(
        classifier,
        train_feature_matrix,
        val_feature_matrix,
        train_labels,
        val_labels,
        **kwargs):
    """
    Trains a linear classifier and computes accuracy.
    The classifier is trained on the train data. The classifier's
    accuracy on the train and validation data is then returned.

    Args:
        classifier - A classifier function that takes arguments
            (feature matrix, labels, **kwargs) and returns (theta, theta_0)
        train_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        val_feature_matrix - A numpy matrix describing the training
            data. Each row represents a single data point.
        train_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the training
            feature matrix.
        val_labels - A numpy array where the kth element of the array
            is the correct classification of the kth row of the validation
            feature matrix.
        **kwargs - Additional named arguments to pass to the classifier
            (e.g. T or L)

    Returns: A tuple in which the first element is the (scalar) accuracy of the
    trained classifier on the training data and the second element is the
    accuracy of the trained classifier on the validation data.
    """
    # Your code here
    result_classifier = classifier(
        train_feature_matrix, train_labels, **kwargs)
    theta = result_classifier[0]
    theta_0 = result_classifier[1]

    train_data_classify = classify(train_feature_matrix, theta, theta_0)
    validate_data_classify = classify(val_feature_matrix, theta, theta_0)

    return accuracy(train_data_classify, train_labels), accuracy(validate_data_classify, val_labels)
    # raise NotImplementedError
# pragma: coderesponse end


# pragma: coderesponse template
def extract_words(input_string):
    """
    Helper function for bag_of_words()
    Inputs a text string
    Returns a list of lowercase words in the string.
    Punctuation and digits are separated out into their own words.
    """
    for c in punctuation + digits:
        input_string = input_string.replace(c, ' ' + c + ' ')

    return input_string.lower().split()
# pragma: coderesponse end


# pragma: coderesponse template
def bag_of_words(texts):
    """
    Inputs a list of string reviews
    Returns a dictionary of unique unigrams occurring over the input

    Feel free to change this code as guided by Problem 9
    """
    stop_words = {}
    with open('stopwords.txt', 'r') as file:
        for word in file.readlines():
            stop_words[word.strip()] = True
    # Your code here
    dictionary = {}  # maps word to unique index
    for text in texts:
        word_list = extract_words(text)
        for word in word_list:
            if word not in dictionary and word not in stop_words:
                dictionary[word] = len(dictionary)

    return dictionary
# pragma: coderesponse end


# pragma: coderesponse template
def extract_bow_feature_vectors(reviews, dictionary):
    """
    Inputs a list of string reviews
    Inputs the dictionary of words as given by bag_of_words
    Returns the bag-of-words feature matrix representation of the data.
    The returned matrix is of shape (n, m), where n is the number of reviews
    and m the total number of entries in the dictionary.

    Feel free to change this code as guided by Problem 9
    """
    # Your code here
    # print('reviews:', reviews)
    # print('dictionary:', dictionary)
    num_reviews = len(reviews)
    feature_matrix = np.zeros([num_reviews, len(dictionary)])

    for i, text in enumerate(reviews):
        word_list = extract_words(text)
        word_count = {}

        for word in word_list:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1

        for word in word_list:
            if word in dictionary:
                feature_matrix[i, dictionary[word]] = word_count[word]
    # print('feature_matrix:', feature_matrix)
    return feature_matrix
# pragma: coderesponse end


# pragma: coderesponse template
def accuracy(preds, targets):
    """
    Given length-N vectors containing predicted and target labels,
    returns the percentage and number of correct predictions.
    """
    return (preds == targets).mean()
# pragma: coderesponse end
