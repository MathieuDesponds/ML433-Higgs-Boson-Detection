import numpy as np
from implementations import *
from helpers import *
from processing import *

path_dataset_train = "./data/train.csv.zip"
path_dataset_test = "./data/test.csv.zip"

y_train, x_train, ids_train = load_data(path_dataset_train)
_, x_test, ids_test = load_data(path_dataset_test)

poly_degree_opti = [11, 3, 6, 3, 9, 3]
lower_q_opti = [1, 0, 0, 6, 3, 0]
higher_q_opti = [98, 98, 96, 96, 91, 98]

percentiles = [lower_q_opti, higher_q_opti]
x_train_l, ids_train_separated, y_l, x_test_l, id_test_l, _ = data_clean_all_sep(
    x_train,
    x_test,
    y_train,
    np.ones(x_test.shape[0]),
    ids_train,
    ids_test,
    percentiles,
    poly_degree_opti,
)

w_l = []
threshold_gd = 1e-8

################################ 0 ################################
x, y = x_train_l[0], y_l[0]
w_gd, loss = ridge_regression(y, x, 0.000001)
w_l.append(w_gd)

################################ 1 ################################
x, y = x_train_l[1], y_l[1]
w_gd, _ = reg_logistic_regression(
    y, x, 0.000001, np.ones((x.shape[1], 1)), 350, 1, threshold_gd, stochastic=False
)
w_l.append(w_gd)

################################ 2 ################################
x, y = x_train_l[2], y_l[2]
w_gd, _ = ridge_regression(y, x, 0.000005)
w_l.append(w_gd)

################################ 3 ################################
x, y = x_train_l[3], y_l[3]
w_gd, _ = least_squares(y, x)
w_l.append(w_gd)

################################ 4 ################################
x, y = x_train_l[4], y_l[4]
w_gd, _ = ridge_regression(y, x, 0.000001)
w_l.append(w_gd)

################################ 5 ################################
x, y = x_train_l[5], y_l[5]
w_gd, _ = reg_logistic_regression(
    y, x, 0.000001, np.ones((x.shape[1], 1)), 310, 1, threshold_gd, stochastic=False
)
w_l.append(w_gd)

y_test_pred_list = eval_l(w_l, x_test_l, 0.5 * np.ones(6), False)

y_pred = np.concatenate(
    (
        y_test_pred_list[0],
        y_test_pred_list[1],
        y_test_pred_list[2],
        y_test_pred_list[3],
        y_test_pred_list[4],
        y_test_pred_list[5],
    )
)
ids_of_test = np.concatenate(
    (id_test_l[0], id_test_l[1], id_test_l[2], id_test_l[3], id_test_l[4], id_test_l[5])
)

create_csv_submission(ids_of_test, y_pred, "result.csv")
