import numpy as np
import random

"""
# This file contains all these functions 

### Cost functions
    ### compute_mse(y, tx, w)
    ### compute_rmse(y,tx,w)
### Gradient descent
    ### compute_mean_squared_gradient(y, tx, w)
    ### mean_squared_error_gd(y, tx, initial_w, max_iters, gamma)
    ### mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma)
    
### Least Squares 
    ### least_squares(y, tx)

### Ridge regression 
    ### ridge_regression(y, tx, lambda_)

### Logistic Regression
    ### logistic_regression(y, tx, initial_w, max_iter, gamma, threshold=1e-8, stochastic=False)

### Regularized Logistic Regression
    ### reg_logistic_regression(y, tx, lambda_, initial_w, max_iter, gamma, threshold=1e-8, stochastic=False)
"""

########################################
###         Cost functions           ###
########################################


def compute_mse(y, tx, w):
    """Calculate the loss using MSE.

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N,D)
        w: numpy array of shape=(D, 1). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    err = np.subtract(y, tx.dot(w))
    return (0.5 / y.shape[0] * err.T.dot(err))[0][0]


def compute_rmse(y, tx, w):
    """Calculate the loss using RMSE.

    Args:
        y: numpy array of shape (N, 1)
        tx: numpy array of shape (N, 2)
        w: numpy array of shape (D, 1). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    return np.sqrt(2 * compute_mse(y, tx, w))


########################################
###        Gradient descent          ###
########################################


def compute_mean_squared_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape (N, 1)
        tx: numpy array of shape (N, D)
        w: numpy array of shape (D, 1). The vector of model parameters.

    Returns:
        An numpy array of shape (D, 1) (same shape as w), containing the gradient of the loss at w.
    """
    N = y.shape[0]
    err = y - tx.dot(w)
    gradient = -1 / N * tx.T.dot(err)
    return gradient


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape (N, 1)
        tx: numpy array of shape (N, D)
        initial_w: numpy array of shape (D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the model parameter a as numpy array of shape (D, 1), for last iter of GD
        loss: the MSE loss value (scalar) of the last iteration of GD
    """
    # Define parameters to store w and loss
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_mean_squared_gradient(y, tx, w)
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD) with batch size = 1.

    Args:
        y: numpy array of shape (N, 1)
        tx: numpy array of shape (N, 2)
        initial_w: numpy array of shape (D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the model parameter a as numpy array of shape (D, 1), for last iter of SGD
        loss: the MSE loss value (scalar) of the last iteration of SGD
    """
    n = y.shape[0]
    w = initial_w
    for n_iter in range(max_iters):
        idx = np.random.randint(0, n)
        gradient = compute_mean_squared_gradient(
            np.array([y[idx]]), np.array([tx[idx]]), w
        )
        w = w - gamma * gradient
    loss = compute_mse(y, tx, w)
    return w, loss


########################################
###          Least squares           ###
########################################


def least_squares(y, tx):
    """Calculate the least squares solution.

    Args:
        y: numpy array of shape (N,  1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.

    Returns:
        w: optimal weights, numpy array of shape(D, 1), D is the number of features.
        mse: scalar.

    """
    w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    mse = compute_mse(y, tx, w)
    return w, mse


########################################
###        Ridge Regression          ###
########################################
def ridge_regression(y, tx, lambda_):
    """Implement ridge regression.

    Args:
        y: numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N ,D), D is the number of features.
        lambda_: a scalar denoting the penalty value

    Returns:
        w: optimal weights, numpy array of shape(D, 1), D is the number of features.
        loss: the loss value (scalar) of the optimal weight
    """
    n = y.shape[0]
    d = tx.shape[1]
    w = np.linalg.solve(tx.T.dot(tx) + lambda_ * (2 * n) * np.eye(d), tx.T.dot(y))
    loss = compute_mse(y, tx, w)
    return w, loss


########################################
###       Logistic regression        ###
########################################


def sigmoid(t):
    """Apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """Compute the  loss.

    Args:
        y:  numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w:  numpy array of shape (D, 1). The vector of model parameters.

    Returns:
        loss: a non-negative loss
    """

    N = y.shape[0]
    tx_w = tx.dot(w)
    loss = 1 / N * np.sum(np.log(1 + np.exp(tx_w)) - y * tx_w)
    return loss


def calculate_gradient(y, tx, w, stochastic=False):
    """Compute the gradient of loss.

    Args:
        y:  numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w:  numpy array of shape (D, 1). The vector of model parameters.
        stochastic : True if we use sgd

    Returns:
        gradient: a vector of shape (D, 1)
    """
    assert y.shape[0] == tx.shape[0]
    assert tx.shape[1] == w.shape[0]
    # ***************************************************
    N = y.shape[0]
    tx_w = tx.dot(w)
    if stochastic:
        idx = random.randint(0, N - 1)
        gradient = np.array([tx[idx]]).T * (sigmoid(tx[idx].dot(w)) - y[idx])
    else:
        gradient = 1 / N * tx.T.dot(sigmoid(tx_w) - y)
    return gradient


def learning_by_gradient_descent(y, tx, w, gamma, stochastic=False):
    """Do one step of gradient descent using logistic regression. Return the loss and the updated w.

    Args:
        y:  numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w:  numpy array of shape (D, 1). The vector of model parameters.
        gamma: a scalar denoting the stepsize
        stochastic : True if we use sgd

    Returns:
        loss: scalar number
        w: numpy array of shape (D, 1). The updated vector of model parameters.
    """
    # ***************************************************
    loss = calculate_loss(y, tx, w)
    gradient = calculate_gradient(y, tx, w, stochastic)
    w = w - gamma * gradient
    return loss, w


def logistic_regression(
    y, tx, initial_w, max_iter, gamma, threshold=1e-8, stochastic=False
):
    """Implement logistic regression. Return the loss and the w.

    Args:
        y:  numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        initial_w:  numpy array of shape (D, 1). The vector of model parameters.
        max_iter: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize
        threshold: determining the stop condition
        stochastic: True if we use sgd

    Returns:
        loss: scalar number
        w: numpy array of shape (D, 1).
    """
    w = initial_w
    loss_last, loss_new = 100, -100
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss_last = loss_new
        loss_new, w = learning_by_gradient_descent(y, tx, w, gamma, stochastic)
        # check if we have to stop
        if np.abs(loss_last - loss_new) < threshold:
            break
    loss = calculate_loss(y, tx, w)
    return w, loss


########################################
### Regularized logistic regression  ###
########################################
def penalized_logistic_regression(y, tx, w, lambda_, stochastic):
    """return the loss and gradient.

    Args:
        y:  numpy array of shape (N, 1), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
        w:  numpy array of shape (D, 1). The vector of model parameters.
        lambda_: a scalar denoting the penalty value
        stochastic: True if we use sgd

    Returns:
        loss: scalar number
        gradient: shape=(D, 1)
    """

    loss = calculate_loss(y, tx, w) + lambda_ * w.T.dot(w)
    gradient = calculate_gradient(y, tx, w, stochastic) + 2 * lambda_ * w
    return loss, gradient


def learning_by_penalized_gradient(y, tx, w, gamma, lambda_, stochastic):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
        gamma: a scalar denoting the stepsize
        lambda_: a scalar denoting the penalty value
        stochastic: True if we use sgd

    Returns:
        loss: scalar number
        w: shape=(D, 1)
    """
    loss, gradient = penalized_logistic_regression(y, tx, w, lambda_, stochastic)
    w = w - gamma * gradient
    return loss, w


def reg_logistic_regression(
    y, tx, lambda_, initial_w, max_iter, gamma, threshold=1e-8, stochastic=False
):
    """The regularized logistic regression algorithm.

    Args:
        y: numpy array of shape=(N, 1)
        tx: numpy array of shape=(N, D)
        initial_w: numpy array of shape=(D, 1). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations
        gamma: a scalar denoting the stepsize
        threshold: determining the stop condition
        stochastic: determining if we use stochastic regularized logistic regression


    Returns:
        w: the model parameter a as numpy array of shape (D, 1), for last iter of GD
        loss: the MSE loss value (scalar) of the last iteration of GD
    """

    w = initial_w
    losses = []
    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_, stochastic)
        losses.append(loss)
        # check if we have to stop
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    loss = calculate_loss(y, tx, w)
    return w, loss
