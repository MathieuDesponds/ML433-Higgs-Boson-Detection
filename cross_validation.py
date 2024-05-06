import numpy as np
import matplotlib.pyplot as plt


from implementations import *
from processing import *
from helpers import *

# ***************************************************
# Grid search with cross validation
# ***************************************************
def grid_search_with_cross_validation(
    y_train,
    x_train,
    ids_train,
    nb_division_test,
    poly,
    min_qs,
    max_qs,
    k_fold,
    f_train,
    lambdas=[-1],
    gammas=[-1],
):
    """
    For a given training function and a set of parameters we want to test, train the model and print the best parameters for each subdivision of the dataset

    Args :
        y_train : numpy array of shape=(N, 1) the class labels of the training set
        x_train : numpy array of shape=(N,D) the features of the training set
        ids_train : numpy array of shape=(N, 1) event ids of the training set

        nb_division_test : a scalar with how many separation we have on our dataset
        poly : an array of degree for the polynomial expension
        min_qs : an array of min quantils that we will use to min bring outliers in a better range
        max_qs : an array of min quantils that we will use to min bring outliers in a better range
        k_fold : scalar the number of fold for the cross validation
        f_train : a lambda, the training function
        lambdas = [-1] : an array of lambdas we want to test if used in f_train,
        gammas = [-1] : an array of gammas we want to test if used in f_train

    Returns:
        result_loss : a array of size (nb_division_test, len(poly), len(min_qs), len(max_qs)) with the min loss over all lambdas and gammas for these parameters
        best_params : a array of size (nb_division_test, len(poly), len(min_qs), len(max_qs)) with the ids of the parameters of the min loss for each nb_division_test
    """
    result_loss = np.zeros((nb_division_test, len(poly), len(min_qs), len(max_qs)))
    result_lambda = np.zeros((nb_division_test, len(poly), len(min_qs), len(max_qs)))
    result_gamma = np.zeros((nb_division_test, len(poly), len(min_qs), len(max_qs)))
    result_ws = np.zeros(
        (nb_division_test, len(poly), len(min_qs), len(max_qs)), dtype=object
    )

    for p_i, p in enumerate(poly):
        for min_i, min_q in enumerate(min_qs):
            for max_i, max_q in enumerate(max_qs):
                print(
                    "Polynomial expension : %d, min_Q : %d, max_Q : %d"
                    % (p, min_q, max_q)
                )
                # load the data according to the chosen parameters
                x_train_l, _, y_l = data_clean(
                    x_train,
                    ids_train,
                    y_train,
                    bring_outlier=[min_q, max_q],
                    poly_exp_deg=p,
                )
                # For each of the subset train the model and find the best parameters with the above parameters
                for test in range(nb_division_test):
                    x, y = x_train_l[test], y_l[test]
                    initial_w = np.ones(np.shape(x_train_l[test])[1])
                    best_lambda, best_gamma, best_mse, best_w = choose_lambda_gamma(
                        y, x, lambdas, k_fold, f_train, gammas, initial_w, seed=12
                    )
                    result_loss[test, p_i, min_i, max_i] = best_mse
                    result_lambda[test, p_i, min_i, max_i] = best_lambda
                    result_gamma[test, p_i, min_i, max_i] = best_gamma
                    result_ws[test, p_i, min_i, max_i] = best_w
    # After having tested all params find the best loss for each of the subdivisions
    best_params = []
    y_test_pred_list = []
    for test in range(nb_division_test):
        min_idx = np.unravel_index(
            np.nanargmin(result_loss[test]), result_loss[test].shape
        )
        best_params.append(min_idx)
        best_p_i, best_min_q_i, best_max_q_i = min_idx[0], min_idx[1], min_idx[2]
        best_w_overall = result_ws[test, best_p_i, best_min_q_i, best_max_q_i]
        # Reload the data with the bests params
        x_train_l, _, y_l = data_clean(
            x_train, ids_train, y_train, bring_outlier=[min_q, max_q], poly_exp_deg=p
        )

        # Compute its accuracy and print the results
        y_test_pred_list.append(
            give_labels(x_train_l[test], best_w_overall, 0.5, zero=False)
        )
        acc = compute_accuracy(y_test_pred_list[test], y_l[test])
        print(
            "Overall for test %d the best score is %7f with param \n poly : %3d, quantile_low : %3d, quantile_high : %3d"
            % (test, acc, poly[best_p_i], min_qs[best_min_q_i], max_qs[best_max_q_i])
            + (
                ", lambda : %7f"
                % (result_lambda[test, best_p_i, best_min_q_i, best_max_q_i])
                if lambdas[0] != -1
                else ""
            )
            + (
                ", gamma : %7f"
                % (result_gamma[test, best_p_i, best_min_q_i, best_max_q_i])
                if gammas[0] != -1
                else ""
            )
        )

    return result_loss, best_params


def choose_lambda_gamma(y, x, lambdas, k_fold, f_train, gammas, initial_w, seed=12):
    """
    For a given training function and a set of parameters we want to test, train the model with the lambdas and gammas given and return th best ones

    Args :
        y:          numpy array of shape=(N, 1) the class labels
        x :         numpy array of shape=(N,D) the features

        lambdas :   an array of lambdas we want to test if used in f_train,
        k_fold :    scalar the number of fold for the cross validation
        f_train :   a lambda, the training function
        gammas  :   an array of gammas we want to test if used in f_train
        initial_w : the initial_w if needed for the training function
        seed :      a seed for reproducibility

    Returns:
        best_lambda : a scalar, lambda for the min loss
        best_gamma : a scalar, gamma for the min loss,
        best_mse : the best loss
        best_w : the model that gave the best loss
    """
    # split data in k fold
    k_indices = build_k_indices(y, k_fold, seed)
    # define lists to store the loss of training data and test data
    mse_tr = np.zeros((len(gammas), len(lambdas)))
    mse_te = np.zeros((len(gammas), len(lambdas)))
    wss = np.zeros((len(gammas), len(lambdas)), dtype=object)
    # ***************************************************
    for i, gamma in enumerate(gammas):
        for j, lambda_ in enumerate(lambdas):
            loss_sum_tr, loss_sum_te = 0, 0
            ws = np.zeros((x.shape[1], 1))
            for k in range(k_fold):
                # Make a CV on the kth fold
                loss_tr, loss_te, w = cross_validation(
                    y, x, f_train, k_indices, k, lambda_, gamma
                )
                loss_sum_tr += loss_tr
                loss_sum_te += loss_te
                ws += w
            mse_tr[i][j] = loss_sum_tr / k_fold
            mse_te[i][j] = loss_sum_te / k_fold
            wss[i][j] = ws / k_fold
    if np.isnan(mse_te).all():
        return -1, -1, 1000000, initial_w
    min_idx = np.unravel_index(np.nanargmin(mse_te), mse_te.shape)
    best_gamma = gammas[min_idx[0]]
    best_lambda = lambdas[min_idx[1]]
    best_mse = mse_te[min_idx[0]][min_idx[1]]
    best_w = wss[min_idx[0]][min_idx[1]]
    # ***************************************************

    # cross_validation_visualization(lambdas, mse_tr, mse_te)
    # print("The choice of lambda which leads to the best test mse is %.5f with a test mse of %.3f" % (best_lambda, best_mse))
    return best_lambda, best_gamma, best_mse, best_w


def cross_validation(y, x, f_train, k_indices, k, lambda_, gamma):
    """return the loss of ridge regression for a fold corresponding to k_indices

    Args:
        y:          shape=(N,) the class labels
        x:          shape=(N, D) the features
        f_train :   a lambda, the training function
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar
        gammas  :   scalar

    Returns:
        loss_tr :   the loss on the training set
        loss_te :   the loss on the test set
        w       :   the model we got
    """
    # get k'th subgroup in test, others in train
    idx_test, idx_train = (
        k_indices[k],
        np.concatenate(k_indices[np.arange(len(k_indices)) != k]),
    )
    x_test, y_test = x[idx_test], y[idx_test]
    x_train, y_train = x[idx_train], y[idx_train]

    w, loss = f_train(y_train, x_train, lambda_, gamma, np.ones((x_train.shape[1], 1)))
    # calculate the loss for train and test data
    loss_tr, loss_te = compute_mse(y_train, x_train, w), compute_mse(y_test, x_test, w)
    return loss_tr, loss_te, w


def statistics_on_best_params(
    x_train,
    ids_train,
    y_train,
    lambdas,
    nb_iters,
    k_fold,
    f_train,
    gammas,
    poly_exp,
    min_qs,
    max_qs,
):
    """
    For a given training function and a set of parameters we want to test, train the model with the lambdas and gammas given and return th best ones

    Args :
        y:          numpy array of shape=(N, 1) the class labels
        x :         numpy array of shape=(N,D) the features

        lambda :   lambda we want to test if used in f_train,
        nb_iters : the number of time we want to repeat the experience
        k_fold :    scalar the number of fold for the cross validation
        f_train :   a lambda, the training function
        gamma  :   gamma we want to test if used in f_train
        initial_w : the initial_w if needed for the training function
        seed :      a seed for reproducibility

    Returns:
        losses_tr:  All the losses of the training set
        losses_te:  All the losses of the test set
        ws :        All the models we got in the cross validaiton
    """
    x_train_separated, ids_train_separated, y_separated = data_clean_sep(
        x_train,
        ids_train,
        y_train,
        bring_outlier=[min_qs, max_qs],
        poly_exp_deg=poly_exp,
    )
    # define lists to store the loss of training data and test data
    losses_tr, losses_te, ws = [], [], []
    # split data in k
    for iter in range(nb_iters):
        all_k_indices = []
        for test in range(len(x_train_separated)):
            all_k_indices.append(
                build_k_indices(y_separated[test], k_fold, np.random.randint(100000))
            )
        loss_tr_sum, loss_te_sum = 0, 0
        # w_sum = np.zeros((x_train_separated[test].shape[1],1))
        for k in range(k_fold):
            for test in range(len(x_train_separated)):
                # Make a CV on the kth fold for separation 'test'
                loss_tr, loss_te, w = cross_validation(
                    y_separated[test],
                    x_train_separated[test],
                    f_train,
                    all_k_indices[test],
                    k,
                    lambdas[test],
                    gammas[test],
                )
                loss_tr_sum += loss_tr * y_separated[test].shape[0]
                loss_te_sum += loss_te * y_separated[test].shape[0]
        losses_tr.append(loss_tr_sum / (k_fold * y_train.shape[0]))
        losses_te.append(loss_te_sum / (k_fold * y_train.shape[0]))

    return losses_tr, losses_te, ws


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold.

    Args:
        y:      shape=(N,)
        k_fold: K in K-fold, i.e. the fold num
        seed:   the random seed

    Returns:
        A 2D array of shape=(k_fold, N/k_fold) that indicates the data indices for each fold
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval : (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation_visualization(lambds, rmse_tr, rmse_te):
    """visualization the curves of rmse_tr and rmse_te."""
    plt.semilogx(lambds, rmse_tr, marker=".", color="b", label="train error")
    plt.semilogx(lambds, rmse_te, marker=".", color="r", label="test error")
    plt.xlabel("lambda")
    plt.ylabel("r mse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
