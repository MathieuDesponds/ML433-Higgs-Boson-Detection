import numpy as np
import matplotlib.pyplot as plt
import csv
from zipfile import ZipFile


from implementations import *
from processing import *

########################################
###            Helpers               ###
########################################


def load_data(data_path, sub_sample=False):
    """Loads data and returns y (class labels), x (features) and ids (event ids)
    Args:
        data_path: a string containing the path of the file
        sub_sample : True if we want sub sample

    Returns:
        y : the class labels
        x : the features
        ids : event ids
    """

    with ZipFile(data_path) as zf:
        prediction = np.genfromtxt(
            zf.open(data_path[7:-4]), delimiter=",", skip_header=1, dtype=str, usecols=1
        )
        data = np.genfromtxt(zf.open(data_path[7:-4]), delimiter=",", skip_header=1)
    ids = data[:, 0].astype(int)
    x = data[:, 2:]

    y = np.ones(len(prediction))
    y[
        np.where(prediction == "b")
    ] = 0  # convert labels from ('b', 's') to binary (0, 1)

    # sub-sample
    if sub_sample:
        y = y[::50]
        x = x[::50]
        ids = ids[::50]

    return y, x, ids


def load_all_data():
    """Loads the training and the test sets
    Args:

    Returns:
        y_train : numpy array of shape=(N, 1) the class labels of the training set
        x_train : numpy array of shape=(N,D) the features of the training set
        ids_train : numpy array of shape=(N, 1) event ids of the training set
        x_test : numpy array of shape=(N,D) the features of the test set
        ids_test : numpy array of shape=(N, 1) event ids of the test set
    """
    path_dataset_train = "./data/train.csv.zip"
    path_dataset_test = "./data/test.csv.zip"

    y_train, x_train, ids_train = load_data(path_dataset_train)
    _, x_test, ids_test = load_data(path_dataset_test)

    return y_train, x_train, ids_train, x_test, ids_test


def give_labels(data, w, threshold, zero=True):
    """Give labels in function of the weights and a threshold
    Args:
        data : numpy array of shape=(N,D) all the features
        w : numpy array of shape=(D,1) the model we trained
        threshold : scalar the value that separate the two classes
    Returns:
        labels : numpy array of shape=(N,1) the class labels for the data and the chosen model
    """
    y = np.dot(data, w)
    y[np.where(y <= threshold)] = 0 if zero else -1
    y[np.where(y > threshold)] = 1
    return y


def compute_accuracy(y_pred, y):
    """Compute the accuracy (i.e. the ratio of good predictions)"""
    score = np.count_nonzero(y == y_pred)
    return score / y_pred.shape[0]


def create_csv_submission(ids, y_pred, name):
    """
    Creates an output file in .csv format for submission to Kaggle or AIcrowd
    Arguments:
            ids (event ids associated with each prediction)
            y_pred (predicted class labels)
            name (string name of .csv output file to be created)
    """
    with open(name, "w") as csvfile:
        fieldnames = ["Id", "Prediction"]
        writer = csv.DictWriter(csvfile, delimiter=",", fieldnames=fieldnames)
        writer.writeheader()
        for r1, r2 in zip(ids, y_pred):
            writer.writerow({"Id": int(r1), "Prediction": int(r2)})


def eval_l(w_list, x_test_l, threshold=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], zero=True):
    """
    Output the prediction (0 or 1) given the weights, the features and a threshold
    Arguments: w_list (list of weights for each split)
               x_test_l (list of the split data)
               threshold (list of threshold)
               zero (Boolean) if we want 0 or -1 for background
    """
    y_pred_list = []
    for i in range(len(w_list)):
        y_pred_list.append(
            give_labels(x_test_l[i], w_list[i].flatten(), threshold[i], zero)
        )

    return y_pred_list
