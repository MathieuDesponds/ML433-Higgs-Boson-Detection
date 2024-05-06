import numpy as np
from zipfile import ZipFile

from helpers import *


# The nan features conditioned on the feature 22
nan_features = [
    [4, 5, 6, 7, 8, 12, 15, 16, 18, 20, 22, 23, 24, 25, 26, 27, 28, 29],
    [4, 5, 6, 7, 8, 12, 15, 16, 18, 20, 22, 25, 26, 27, 28],
    [7, 8, 15, 16, 18, 20, 22, 25, 26, 28],
    [7, 8, 15, 16, 18, 20, 22, 25, 26, 28],
]

nan_features6 = [
    [4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
    [0, 4, 5, 6, 12, 22, 23, 24, 25, 26, 27, 28, 29],
    [4, 5, 6, 12, 22, 26, 27, 28],
    [0, 4, 5, 6, 12, 22, 26, 27, 28],
    [22],
    [0, 22],
]

# Which features we want to take the log or the log log of
# log_f = [0, 1, 2, 3, 4, 5, 7, 8, 10, 19, 21, 29]
log_f = [4, 5, 9, 21, 23]

log_log_f = []
# log_log_f = [9, 13, 16, 23, 26]
log_f_plots = [
    [0, 2, 5, 6, 15],
    [2, 4, 8],
    [0, 2, 6, 8, 9, 10, 13, 18, 19, 21],
    [1, 2, 7, 8, 11, 14, 18],
    [4, 5, 6, 8, 11, 12, 13, 16, 22, 23, 25, 26, 28],
    [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 15, 22, 24, 25],
]
log_log_f_plots = [[3, 4, 8], [1, 7], [4], [0, 3, 4], [19], [0, 8, 27]]
log_f_after = [
    [0, 1, 2, 3, 4, 5, 7, 15, 17],
    [0, 1, 2, 3, 4, 6, 14, 16],
    [0, 1, 2, 3, 4, 5, 7, 15, 17, 21],
    [0, 1, 2, 3, 4, 6, 14, 16, 20],
    [0, 1, 2, 3, 4, 5, 7, 8, 10, 19, 21, 28],
    [0, 1, 2, 3, 4, 6, 7, 9, 18, 20, 27],
]
log_log_f_after = [
    [6, 9, 12],
    [5, 8, 11],
    [6, 9, 12, 18],
    [5, 8, 11, 17],
    [9, 13, 16, 22, 25],
    [8, 12, 15, 21, 24],
]

poly_degree_opti = [6, 4, 5, 2, 8, 2]
lower_q_opti = [0, 6, 0, 3, 3, 10]
higher_q_opti = [96, 98, 96, 97, 93, 94]

percentiles = [lower_q_opti, higher_q_opti]


def standardize(x):
    """
    Standardize the data by column
    Arguments: x shape=(N,D): set to standardize
               x_tr shape=(N',D): training set
    """
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(x, axis=0)
    return std_data


def data_clean(x, ids, y, bring_outlier=percentiles, poly_exp_deg=poly_degree_opti):
    """Clean the data using the same poly for all and the same quantiles and divide the dataset into 6 separations
    Args:
        x : the features
        ids : event ids
        y : the class labels
        bring_outlier : the outlier quantile we want to use
        poly_exp_deg : the degree for the polynomial expension
    Returns:
        x_separated : a array of size 6 with all the features
        ids_separated : a array of size 6 with all the ids
        y_separated :a array of size 6 with all the labels
    """
    xt = x.copy()

    x_separated, ids_separated, y_separated = separate_dataset_in_6(xt, ids, y)
    x_separated = delete_NaN_features(x_separated, nan_features6)

    for test in range(len(x_separated)):
        y_separated[test] = y_separated[test].reshape((len(y_separated[test]), 1))

    for i in range(len(x_separated)):
        bring_outliers_in_range(
            x_separated[i],
            range(0, x_separated[i].shape[1]),
            bring_outlier[0],
            bring_outlier[1],
        )
        x_separated[i] = x_separated[i] - np.mean(x_separated[i], axis=0)
        x_separated[i] = poly_expansion(x_separated[i], int(poly_exp_deg), cross=False)
        x_separated[i] = standardize(x_separated[i])
        x_separated[i] = add_bias(x_separated[i])

    return x_separated, ids_separated, y_separated


def data_clean_sep(x, ids, y, bring_outlier, poly_exp_deg):
    """Divide the dataset into 6 separations and clean the data using the poly and quantiles for each dataset
    Args:
        x : the features
        ids : event ids
        y : the class labels
        bring_outlier : the outlier quantile we want to use
        poly_exp_deg : the degree for the polynomial expension
    Returns:
        x_separated : a array of size 6 with all the features
        ids_separated : a array of size 6 with all the ids
        y_separated :a array of size 6 with all the labels
    """
    xt = x.copy()

    x_separated, ids_separated, y_separated = separate_dataset_in_6(xt, ids, y)
    for test in range(len(x_separated)):
        y_separated[test] = y_separated[test].reshape((len(y_separated[test]), 1))
    x_separated = delete_NaN_features(x_separated, nan_features6)

    for i in range(len(x_separated)):
        bring_outliers_in_range(
            x_separated[i],
            range(0, x_separated[i].shape[1]),
            bring_outlier[0][i],
            bring_outlier[1][i],
        )
        x_separated[i] = x_separated[i] - np.mean(x_separated[i], axis=0)
        x_separated[i] = poly_expansion(x_separated[i], poly_exp_deg[i], cross=False)
        x_separated[i] = standardize(x_separated[i])
        x_separated[i] = add_bias(x_separated[i])

    return x_separated, ids_separated, y_separated


def data_clean_all(
    x_train,
    x_test,
    y_list,
    ids_train,
    ids_test,
    bring_outlier=[10, 90],
    poly_exp_deg=2,
):
    """
    clean all data and apply the same bring outlier and poly exp to all separation
    """
    x_train_separated, ids_train_separated, y_separated = data_clean(
        x_train,
        ids_train,
        y_list,
        bring_outlier=bring_outlier,
        poly_exp_deg=poly_exp_deg,
    )
    x_test_separated, ids_test_separated, _ = data_clean(
        x_test,
        ids_test,
        np.zeros((x_test.shape[0], 1)),
        bring_outlier=bring_outlier,
        poly_exp_deg=poly_exp_deg,
    )
    return (
        x_train_separated,
        ids_train_separated,
        y_separated,
        x_test_separated,
        ids_test_separated,
    )


def data_clean_all_sep(
    x_train, x_test, y_list, t_l, ids_train, ids_test, bring_outlier, poly_exp_deg
):
    """
    clean all data and separate the bring outlier and poly exp depending on the separation
    """
    x_train_separated, ids_train_separated, y_separated = data_clean_sep(
        x_train, ids_train, y_list, bring_outlier, poly_exp_deg
    )
    x_test_separated, ids_test_separated, t_l_separated = data_clean_sep(
        x_test, ids_test, t_l, bring_outlier, poly_exp_deg
    )
    return (
        x_train_separated,
        ids_train_separated,
        y_separated,
        x_test_separated,
        ids_test_separated,
        t_l_separated,
    )


def data_transformations(x, nb_col, lambda_, replace=False):
    """Transform a feature with a given function. It can replace the feature or add a new one.

    Args:
        x : Samples (N, D)
        nb_col : list of the column we want to perform the transformation
        lambda_ : function to apply
        replace (bool, optional): True if we want to replace the feature with the transform.
                                False if we want to add a new feature with the transform.
                                Defaults to False.

    Returns:
        x: samples data after the transformation
    """
    if replace:
        x[:, nb_col] = np.where(
            x[:, nb_col] == -999, x[:, nb_col], lambda_(x[:, nb_col])
        )
    else:
        x = np.c_[
            x, np.where(x[:, nb_col] == -999, x[:, nb_col], lambda_(x[:, nb_col]))
        ]
    return x


def add_bias(x):
    """Add a column of 1

    Args:
        x : Samples (N, D)

    Returns:
        x: Samples (N, D+1)
    """
    return np.c_[np.ones(x.shape[0]), x]


def poly_expansion(x, degree, cross=True):
    """Polynomial expansion of the data

    Args:
        x : Samples (N, D)
        degree : degree of the expansion
        cross (optional): add all the cross terms of the expansion ([a,b] => [a,b,a^2,b^2,ab]).
                        Defaults to True.

    Returns:
        x_exp: samples data with expansion
    """
    if cross:
        start = 0
        x_exp = x
        for k in range(degree - 1):
            temp_x = x_exp
            next_start = x_exp.shape[1]
            for i in range(x.shape[1]):
                x_exp = np.c_[
                    x_exp, (temp_x[:, range(start + i, temp_x.shape[1])].T * x[:, i]).T
                ]
            start = next_start
    else:
        x_exp = x.copy()
        for deg in range(degree):
            x_exp = np.c_[x_exp, np.power(x, deg + 2)]

    return x_exp


def bring_outliers_in_range(x, feature_nb, lower_bound, upper_bound):
    """Bring the outliers of each feature in a certain range.

    Args:
        x : Samples (N, D)
        feature_nb : list of the features we bring in the range
        lower_bound : lower quantile (must be [0-100])
        upper_bound : upper quantile (must be [0-100])

    Returns:
        x: modified samples data
    """

    l = np.array([feature_nb]).flatten()

    q_low = np.percentile(x, lower_bound)
    q_high = np.percentile(x, upper_bound)

    x[:, feature_nb] = np.where(x[:, feature_nb] <= q_low, q_low, x[:, feature_nb])
    x[:, feature_nb] = np.where(x[:, feature_nb] >= q_high, q_high, x[:, feature_nb])

    return x


def separate_dataset_in_6(x, ids, y=None):
    """Separate the dataset in 6 dataset with the feature 22 and the value of feature 0 (nan or not)

    Args:
        x : Data in an array (N,D)
        ids : ids of the samples
        y (optional): label. Defaults to None.

    Returns:
        x_separated: list with the 6 segments of the data
        ids_separated: list with the ids of the 6 segments
        y_separated: list with the 6 segments of label
    """

    x_separated = []
    y_separated = []
    ids_separated = []
    for i in range(3):
        if i == 2:  # We group the 2 and 3 jets together
            indices = np.where(np.logical_or(x[:, 22] == i, x[:, 22] == i + 1))
        else:
            indices = np.where(x[:, 22] == i)

        (
            x_with_999,
            ids_with_999,
            x_without_999,
            ids_without_999,
            y_with_999,
            y_without_999,
        ) = split_using_f0(x[indices], ids[indices], y[indices])
        x_separated.append(x_with_999)
        ids_separated.append(ids_with_999)

        x_separated.append(x_without_999)
        ids_separated.append(ids_without_999)

        if y is not None:
            y_separated.append(y_with_999)
            y_separated.append(y_without_999)
    if y is None:
        return x_separated, ids_separated
    else:
        return x_separated, ids_separated, y_separated


def add_feature_about_f0(x):
    """Add a dummy feature that indicates if feature 0 has -999 value

    Args:
        x : Samples (N, D)

    Returns:
        x: Samples with the new dummy feature
    """
    filter = np.where(x[:, 0] == -999, 1, 0)
    x = np.c_[x, filter]
    return x


def delete_NaN_features(x, indices):
    """Delete the features in function of their id

    Args:
        x : Samples
        indices : list of ids of the features that must be deleted

    Returns:
        x_train_separated_without_NaN: Samples without the corresponding features
    """
    x_train_separated_without_NaN = x
    for j in range(len(x)):
        x_train_separated_without_NaN[j] = np.delete(
            x_train_separated_without_NaN[j], indices[j], axis=1
        )

    return x_train_separated_without_NaN


def modify_NaN_in_f0(x, function):
    """Replace the NaN in feature 0 with the mean or the medium

    Args:
        x : Samples (N, D)
        function (str): "medium" or "mean"

    Returns:
        x: Samples with modified feature 0
    """
    replace_value = 0
    indices = np.where(x[:, 0] != -999)
    x_without_NaN = x[indices, 0].flatten()

    if function == "mean":
        replace_value = np.mean(x_without_NaN)
    elif function == "median":
        replace_value = np.median(x_without_NaN)

    x[:, 0] = np.where(x[:, 0] == -999, replace_value, x[:, 0])

    return x


def split_using_f0(x, ids, y):
    """Separate the data if they have a nan in feature 0"""

    indices_without_999 = np.where(x[:, 0] == -999)
    indices_with_999 = np.where(x[:, 0] != -999)

    x_without_999 = x[indices_without_999]
    x_with_999 = x[indices_with_999]

    ids_without_999 = ids[indices_without_999]
    ids_with_999 = ids[indices_with_999]

    y_with_999 = y[indices_with_999]
    y_without_999 = y[indices_without_999]

    return (
        x_with_999,
        ids_with_999,
        x_without_999,
        ids_without_999,
        y_with_999,
        y_without_999,
    )
