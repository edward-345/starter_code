"""STA314 Homework 3.

Copyright and Usage Information
===============================

This file is provided solely for the personal and private use of students
taking STA314 at the University of Toronto St. George campus. All forms of
distribution of this code, whether as given or with any changes, are
expressly prohibited.
"""


from utils import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def logistic_predict(weights, data):
    """ Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :return: A vector of probabilities with dimension N x 1, which is the output
    to the classifier.
    """
    #####################################################################
    # TODO:                                                             #
    #####################################################################
    N = data.shape[0]
    ones = np.ones((N, 1))

    design = np.hstack((data, ones))

    z = np.dot(design, weights)
    
    y = sigmoid(z)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return y


def evaluate(targets, y):
    """ Compute evaluation metrics.

    Note: N is the number of examples
          D is the number of features per example

    :param targets: A vector of targets with dimension N x 1.
    :param y: A vector of probabilities with dimension N x 1.
    :return: A tuple (ce, frac_correct)
        WHERE
        ce: (float) Averaged cross entropy
        frac_correct: (float) Fraction of inputs classified correctly
    """
    #####################################################################
    # TODO:                                                             #
    #####################################################################
    ce = np.mean(-targets*np.log(y)-(1.0-targets)*np.log(1.0-y))
    
    binary_y = (y >= 0.5)
    frac_correct = np.mean(binary_y == targets)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return ce, frac_correct


def logistic(weights, data, targets, hyperparameters):
    """ Calculate the cost of penalized logistic regression and its derivatives
    with respect to weights. Also return the predictions.

    Note: N is the number of examples
          D is the number of features per example

    :param weights: A vector of weights with dimension (D + 1) x 1, where
    the last element corresponds to the bias (intercept).
    :param data: A matrix with dimension N x D, where each row corresponds to
    one data point.
    :param targets: A vector of targets with dimension N x 1.
    :param hyperparameters: The hyperparameter dictionary.
    :returns: A tuple (f, df, y)
        WHERE
        f: The average of the loss over all data points, plus a penalty term.
           This is the objective that we want to minimize.
        df: (D+1) x 1 vector of derivative of f w.r.t. weights.
        y: N x 1 vector of probabilities.
    """
    y = logistic_predict(weights, data)
    lambd = hyperparameters["weight_regularization"]
    #####################################################################
    # TODO:                                                             #
    #####################################################################
    # Exclude the bias parameter when calculating weight regularization.
    nb_weights = np.delete(weights, -1)
    f = (np.mean(-targets*np.log(y)-(1.0-targets)*np.log(1.0-y)) 
    + (lambd/2)*(np.linalg.norm(nb_weights))**2)
    
    N = data.shape[0]
    ones = np.ones((N, 1))
    design = np.hstack((data, ones))
    zb_weights = np.r_[weights[:-1], [[0.0]]]
    df = np.dot(np.transpose(design),(y-targets))/N +lambd*zb_weights
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return f, df, y


def run_logistic_regression():
    # Load all necessary datasets:
    x_train, y_train = load_train()
    # If you would like to use digits_train_small, please uncomment this line:
    # x_train, y_train = load_train_small()
    x_valid, y_valid = load_valid()
    x_test, y_test = load_test()

    n, d = x_train.shape

    #####################################################################
    # TODO:                                                             #
    # Set the hyperparameters for the learning rate, the number         #
    # of iterations                                                     #
    #####################################################################
    hyperparameters = {
        "learning_rate": 0.21,
        "weight_regularization": 1.0,
        "num_iterations": 2000
    }
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    # Begin learning with gradient descent
    #####################################################################
    # TODO:                                                             #
    # Modify this section to perform gradient descent, create plots,    #
    # compute test error, etc ...                                      #
    #####################################################################
    #Storage
    train_ce_list = []
    val_ce_list = []
    iterations = list(range(1, hyperparameters["num_iterations"] + 1))

    #Training 
    weights = np.zeros((d + 1, 1))
    for t in range(hyperparameters["num_iterations"]):
        f, df, y = logistic(weights, x_train, y_train, hyperparameters)
        weights = weights - hyperparameters["learning_rate"]*df

        train_ce, train_acc = evaluate(y_train, y)
        train_ce_list.append(train_ce)
        
        #Validation for plotting
        pred_y_valid = logistic_predict(weights, x_valid)
        val_ce, val_acc = evaluate(y_valid, pred_y_valid)
        val_ce_list.append(val_ce)
    
    #Plotting Iterations vs Cross Entropy during training 
    plt.plot(iterations, train_ce_list, color = "blue", label = "Training CE")
    plt.plot(iterations, val_ce_list, color = "orange", label = "Validation CE")

    plt.xlabel("Iterations")
    plt.ylabel("Cross Entropy")
    plt.title("Iterations vs Cross Entropy at LR = .21")
    plt.legend()    
    #plt.show()

    #Plotting weight regularization vs Validation CE
    # i manually ran the code after changing the weight regularization each time
    # to get these values. pls check my github to see my progress.
    wr = [0., 0.001, 0.01, 0.1, 1.0]
    valid_ce = [0.0524, 0.0544, 0.0786, 0.1753, 0.4112]

    #Testing
    pred_y_test = logistic_predict(weights, x_test)
    test_ce, test_acc = evaluate(y_test, pred_y_test)

    #print((train_ce, train_acc), (val_ce, val_acc))
    print((val_ce, val_acc),(test_ce, test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    run_logistic_regression()
