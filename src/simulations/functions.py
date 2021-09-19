from itertools import product

import scipy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from numba import jit
from rpy2.robjects.packages import importr
from scipy.stats import iqr
from scipy.stats import skewnorm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.kernel_regression import KernelReg
import lightgbm as lgb


rpy2.robjects.numpy2ri.activate()

# import R's "base" package
base = importr("base")
# import R's "utils" package
utils = importr("utils")
quantregForest = importr("quantregForest")


def tune_nodesize_quantile_forest(X, y, nodesize_grid, pred_band_method, n_simulations):
    result_mat = np.zeros((len(nodesize_grid), n_simulations))

    for j, node_val in enumerate(nodesize_grid):
        for i in range(n_simulations):
            X_train, X_split_again, y_train, y_split_again = train_test_split(
                X, y, test_size=0.4, train_size=0.6
            )
            X_cv, X_calibrate, y_cv, y_calibrate = train_test_split(
                X_split_again, y_split_again, test_size=0.5, train_size=0.5
            )

            # returns array with lower- and upper prediction boundaries for each observation in y_cv
            # pred_band_y_cv = pred_band_method(x_new = X_cv, X_train = X_train, y_train=y_train, X_test=X_calibrate, y_test=y_calibrate, nodesize=node_val)
            pred_band_y_cv = pred_band_method(
                X_predict=X_cv,
                X_train=X_train,
                y_train=y_train,
                X_conf=X_calibrate,
                y_conf=y_calibrate,
                nodesize=node_val,
            )

            length_intervals = pred_band_y_cv[:, 1] - pred_band_y_cv[:, 0]
            criterion = np.mean(length_intervals)
            result_mat[j, i] = criterion

    minimal_average_length_nodesize_index = np.argmin(np.mean(result_mat, axis=1))
    minimal_average_length_nodesize = nodesize_grid[
        minimal_average_length_nodesize_index
    ]
    return minimal_average_length_nodesize, result_mat


def tune_mean_based_rf(X, y, min_samples_leaf_grid, pred_band_method, n_simulations):
    result_mat = np.zeros((len(min_samples_leaf_grid), n_simulations))

    for j, min_samples_leaf_val in enumerate(min_samples_leaf_grid):
        for i in range(n_simulations):
            X_train, X_split_again, y_train, y_split_again = train_test_split(
                X, y, test_size=0.4, train_size=0.6
            )
            X_cv, X_calibrate, y_cv, y_calibrate = train_test_split(
                X_split_again, y_split_again, test_size=0.5, train_size=0.5
            )

            # returns array with lower- and upper prediction boundaries for each observation in y_cv
            pred_band_y_cv = pred_band_method(
                X_predict=X_cv,
                X_train=X_train,
                y_train=y_train,
                X_conf=X_calibrate,
                y_conf=y_calibrate,
                nodesize=min_samples_leaf_val,
            )

            length_intervals = pred_band_y_cv[:, 1] - pred_band_y_cv[:, 0]
            criterion = np.mean(length_intervals)
            result_mat[j, i] = criterion

    minimal_average_length_nodesize_index = np.argmin(np.mean(result_mat, axis=1))
    minimal_average_length_nodesize = min_samples_leaf_grid[
        minimal_average_length_nodesize_index
    ]
    return minimal_average_length_nodesize, result_mat


def numpy_matrix_to_r_matrix(np_mat):
    if len(np_mat.shape) == 1:
        np_mat = np.atleast_2d(np_mat).T
    nr, nc = np_mat.shape
    r_mat = robjects.r.matrix(np_mat, nrow=nr, ncol=nc)
    return r_mat


# 90% coverage, i.e. alpha = 0.1
# still used for cross validation where only predictions for one test set is needed
quantreg_forest = robjects.r(
    """
f_simple <- function(X_train, y_train, X_test, lower, upper, nodesize){
    if(dim(X_train)[2] >= 3){
        mtry = round(dim(X_train)[2] / 3)
    }
    else{
        mtry = 1
    }

    qrf <- quantregForest(x=X_train, y=y_train, nodesize=nodesize, mtry=mtry, ntree=1500)
    conditionalQuantiles  <- predict(object=qrf,  newdata=X_test, what = c(lower, upper))
    return(conditionalQuantiles)
}"""
)


# outdated: used for the conf. inference based on the cumulative density function
quantreg_forest_grid_old = robjects.r(
    """
g_old <- function(X_train, y_train, X_test, alpha_grid, nodesize){
    if(dim(X_train)[2] >= 3){
        mtry = round(dim(X_train)[2] / 3)
    }
    else{
        mtry = 1
    }
    qrf <- quantregForest(x=X_train, y=y_train, nodesize=nodesize, mtry=mtry, ntree=500)
    conditionalQuantiles  <- predict(object=qrf,  newdata=X_test, what = alpha_grid)
    return(conditionalQuantiles)
}"""
)


# used for the conf. inference based on the cumulative density function
quantreg_forest_grid = robjects.r(
    """
g <- function(X_predict, X_train, y_train, X_conf, alpha_grid, nodesize){

    if(dim(X_train)[2] >= 3){
        mtry = round(dim(X_train)[2] / 3)
    }
    else{
        mtry = 1
    }

    qrf <- quantregForest(x=X_train, y=y_train, nodesize=nodesize, mtry=mtry, ntree=500)

    conditionalQuantiles_conf  <- predict(object=qrf, newdata=X_conf, what = alpha_grid)
    conditionalQuantiles_predict <- predict(object=qrf, newdata=X_predict, what = alpha_grid)

    return(list(conditionalQuantiles_conf, conditionalQuantiles_predict))
}"""
)


# used for the conformal inference based on quantile regression
quantreg_forest_beta = robjects.r(
    """
f <- function(X_predict, X_train, y_train, X_conf, lower, upper, nodesize){
    if(dim(X_train)[2] >= 3){
        mtry = round(dim(X_train)[2] / 3)
    }
    else{
        mtry = 1
    }

    qrf <- quantregForest(x=X_train, y=y_train, nodesize=nodesize, mtry=mtry, ntree=500)
    conditionalQuantiles_conf  <- predict(object=qrf, newdata=X_conf, what = c(lower, upper))
    conditionalQuantiles_predict <- predict(object=qrf, newdata=X_predict, what = c(lower, upper))

    return(list(conditionalQuantiles_conf, conditionalQuantiles_predict))
}"""
)


def standardize_data(data):
    scaler = StandardScaler().fit(data)
    std_data = scaler.transform(data)
    return std_data


def standardize_conformal(
    X_train, y_train, X_conformalize, y_conformalize, X_test, y_test
):

    # y_merged = np.concatenate((y_train, y_conformalize, y_test)).reshape(-1, 1)
    # X_merged = np.concatenate((X_train,X_conformalize,X_test), axis= 0)

    X_train_scaler = StandardScaler().fit(X_train)
    # y_train_scaler = MaxAbsScaler().fit(y_train)

    X_train_std = X_train_scaler.transform(X_train)
    X_conformalize_std = X_train_scaler.transform(X_conformalize)
    X_test_std = X_train_scaler.transform(X_test)

    # y_train_std = y_train_scaler.transform(y_train)
    # y_conformalize_std = y_train_scaler.transform(y_conformalize)
    # y_test_std = y_train_scaler.transform(y_test)

    return X_train_std, y_train, X_conformalize_std, y_conformalize, X_test_std, y_test


def split_sample(X, y, test_size):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    return X_train, X_test, y_train, y_test


def OLS(
    X_predict, X_train, y_train, X_conf, intercept=True, fit_training_sample_preds=False
):

    model = LinearRegression(fit_intercept=intercept)
    model.fit(X_train, y_train)

    mu_hat_x_conf = model.predict(X_conf)
    mu_hat_x_predict = model.predict(X_predict)
    mu_hat_x_train = model.predict(X_train)

    return mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf


def random_forest(
    X_predict,
    X_train,
    y_train,
    X_conf,
    n_estimators=1000,
    nodesize=40,
    max_features=None,
):

    if max_features is None:
        if X_train.shape[1] > 1:
            max_features = round(X_train.shape[1] / 3)
        elif X_train.shape[1] == 1:
            max_features = 1
        else:
            raise ValueError("X has a dimensionality problem, missing regressors.")

    model = RandomForestRegressor(
        n_estimators=n_estimators, min_samples_leaf=nodesize, max_features=max_features
    )

    model.fit(X_train, y_train.flatten())

    mu_hat_x_conf = model.predict(X_conf)
    mu_hat_x_predict = model.predict(X_predict)
    mu_hat_x_train = model.predict(X_train)

    return mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf


def kernel_regressor(X_predict, X_train, y_train, X_conf, bandwidth=1, var_type="c"):

    kde_object = KernelReg(
        endog=y_train.flatten(),
        exog=X_train.flatten(),
        var_type=var_type,
        bw=[bandwidth],
    )

    mu_hat_x_train, marg_vals_train = kde_object.fit()
    mu_hat_x_predict, marg_vals_predict = kde_object.fit(X_predict)
    mu_hat_x_conf, marg_vals_conf = kde_object.fit(X_conf)

    return mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf


def absolute_dev_from_mean(y_conf, mu_hat_x_conf):
    return np.abs(y_conf - mu_hat_x_conf)


def weighted_absolute_dev_from_mean(y_conf, mu_hat_x_conf, mad_hat_x_conf):
    return np.abs(y_conf - mu_hat_x_conf) / mad_hat_x_conf


def CQR_conformity_score(lower_quant_hat, upper_quant_hat, y_conf):
    first_arg = lower_quant_hat.flatten() - y_conf.flatten()
    second_arg = y_conf.flatten() - upper_quant_hat.flatten()
    conf_args = np.column_stack((first_arg, second_arg))
    return np.max(conf_args, axis=1)


def pred_band_mean_based(
    X_predict,
    X_train,
    y_train,
    X_conf,
    y_conf,
    algorithm=random_forest,
    alpha=0.1,
    **args
):

    # algorithm used to calculated mu_hat
    mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf = algorithm(
        X_predict=X_predict, X_train=X_train, y_train=y_train, X_conf=X_conf, **args
    )

    conf_scores = absolute_dev_from_mean(
        y_conf=y_conf.flatten(), mu_hat_x_conf=mu_hat_x_conf.flatten()
    )
    k = (1 - alpha) * (1.0 / len(y_conf) + 1)
    d = np.quantile(conf_scores, k)

    pred_band_upper = mu_hat_x_predict + d
    pred_band_lower = mu_hat_x_predict - d

    res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)

    return res


def pred_band_weighted_mean_based(
    X_predict,
    X_train,
    y_train,
    X_conf,
    y_conf,
    algorithm=random_forest,
    alpha=0.1,
    **args
):

    # two step approach for locally weighted split-conformal approach, as described in Lei et al. (2017)

    # algorithm used to calculated mu_hat
    mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf = algorithm(
        X_predict=X_predict, X_train=X_train, y_train=y_train, X_conf=X_conf, **args
    )

    fitted_absolute_residuals_train = np.abs(
        y_train.flatten() - mu_hat_x_train.flatten()
    )

    # algorithm used to calculated mad_hat
    mad_hat_x_predict, mad_hat_x_train, mad_hat_x_conf = algorithm(
        X_predict=X_predict,
        X_train=X_train,
        y_train=fitted_absolute_residuals_train,
        X_conf=X_conf,
        **args
    )

    conf_scores = weighted_absolute_dev_from_mean(
        y_conf=y_conf.flatten(),
        mu_hat_x_conf=mu_hat_x_conf.flatten(),
        mad_hat_x_conf=mad_hat_x_conf.flatten(),
    )

    k = (1 - alpha) * (1.0 / len(y_conf) + 1)
    d = np.quantile(conf_scores, k)

    pred_band_upper = mu_hat_x_predict + mad_hat_x_predict * d
    pred_band_lower = mu_hat_x_predict - mad_hat_x_predict * d

    res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)

    return res


def CV_quantiles_rf(
    X,
    y,
    target_coverage,
    grid_q,
    test_ratio,
    random_state,
    nodesize,
    coverage_factor=0.9,
):

    target_coverage = coverage_factor * target_coverage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )
    best_avg_length = 1e10
    best_q = grid_q[0]

    X_train_r = numpy_matrix_to_r_matrix(X_train)
    X_test_r = numpy_matrix_to_r_matrix(X_test)
    y_train_r = numpy_matrix_to_r_matrix(y_train)

    for q in grid_q:
        preds = np.array(
            quantreg_forest(
                X_train_r,
                y_train_r,
                X_test_r,
                q[0] / 100,
                q[1] / 100,
                nodesize=nodesize,
            )
        )
        coverage, avg_length = compute_coverage_len(y_test, preds[:, 0], preds[:, 1])
        if (coverage >= target_coverage) and (avg_length < best_avg_length):
            best_avg_length = avg_length
            best_q = q
        else:
            break
    return best_q


def compute_coverage_len(y_test, y_lower, y_upper):
    """Compute average coverage and length of prediction intervals
    Parameters
    ----------
    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)
    Returns
    -------
    coverage : float, average coverage
    avg_length : float, average length
    """

    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test) * 100
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage, avg_length


def pred_band_quantile_based_rf(
    X_predict,
    X_train,
    y_train,
    X_conf,
    y_conf,
    coverage_factor=0.9,
    quantiles=[5.0, 95.0],
    CV=True,
    range_vals=15,
    num_vals=5,
    test_ratio=0.1,
    random_state=0,
    nodesize=100,
    alpha=0.1, # this is not used here, instead we use the param "quantiles"
):

    X_train_r = numpy_matrix_to_r_matrix(X_train)
    X_conf_r = numpy_matrix_to_r_matrix(X_conf)
    y_train_r = numpy_matrix_to_r_matrix(y_train)
    X_predict_r = numpy_matrix_to_r_matrix(X_predict)
    alpha = 1.0 - ((quantiles[1] - quantiles[0]) / 100)

    if CV:
        target_coverage = quantiles[1] - quantiles[0]
        grid_q_low = np.linspace(
            quantiles[0], quantiles[0] + range_vals, num_vals
        ).reshape(-1, 1)
        grid_q_high = np.linspace(
            quantiles[1], quantiles[1] - range_vals, num_vals
        ).reshape(-1, 1)
        grid_q = np.concatenate((grid_q_low, grid_q_high), 1)

        cv_quantiles = CV_quantiles_rf(
            X=X_train,
            y=y_train,
            target_coverage=target_coverage,
            grid_q=grid_q,
            test_ratio=test_ratio,
            random_state=random_state,
            coverage_factor=coverage_factor,
            nodesize=nodesize,
        )

        # returns predictions for X_conf and X_predict
        res = np.array(
            quantreg_forest_beta(
                X_predict=X_predict_r,
                X_train=X_train_r,
                y_train=y_train_r,
                X_conf=X_conf_r,
                lower=cv_quantiles[0] / 100.0,
                upper=cv_quantiles[1] / 100.0,
                nodesize=nodesize,
            )
        )

        # res = np.array(quantreg_forest_beta(X_train_r, y_train_r, X_test_r, x_new_r, cv_quantiles[0]/100.0, cv_quantiles[1]/100.0, nodesize=nodesize))

    else:
        res = np.array(
            quantreg_forest_beta(
                X_predict=X_predict_r,
                X_train=X_train_r,
                y_train=y_train_r,
                X_conf=X_conf_r,
                lower=0.05,
                upper=0.95,
                nodesize=nodesize,
            )
        )

    conf_scores = CQR_conformity_score(
        lower_quant_hat=res[0][:, 0], upper_quant_hat=res[0][:, 1], y_conf=y_conf
    )
    k = (1 - alpha) * (1.0 / len(y_conf) + 1)
    d = np.quantile(conf_scores, k)

    pred_band_upper = res[1][:, 1] + d
    pred_band_lower = res[1][:, 0] - d

    result = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)

    return result


@jit(nopython=True)
def conditional_cdf_hat(y_grid, y_vec, q_hat_conf_mat, q_hat_pred_mat):
    # preallocate matrix for the predicted cdf values
    f_hat_y_mat = np.zeros((q_hat_pred_mat.shape[0], len(y_grid.flatten())))

    ###

    q_hat_conf_less_y_mat = q_hat_conf_mat <= y_vec.reshape(-1, 1)
    f_hat_conf = (1.0 / q_hat_conf_less_y_mat.shape[1]) * np.sum(
        q_hat_conf_less_y_mat, axis=1
    )

    ###

    for i, y in enumerate(y_grid):
        q_hat_pred_less_y = q_hat_pred_mat <= y
        f_hat_y = (1.0 / q_hat_pred_less_y.shape[1]) * np.sum(q_hat_pred_less_y, axis=1)

        f_hat_y_mat[:, i] = f_hat_y

    return f_hat_conf, f_hat_y_mat


@jit(nopython=True)
def p_y_func(alpha, y_grid, f_hat_conf, f_hat_y_mat):

    f_hat_conf_abs_dev = np.abs(f_hat_conf.flatten() - 0.5)
    f_hat_y_mat_abs_dev = np.abs(f_hat_y_mat - 0.5)

    conf_set_list = []

    # fix the X_n+1 prediction point:
    for i in range(f_hat_y_mat.shape[0]):
        conf_set = []
        # fix the y grid value:
        for j, y in enumerate(y_grid):
            val = (
                1
                / (len(f_hat_conf_abs_dev) + 1)
                * np.sum(f_hat_y_mat_abs_dev[i, j] <= f_hat_conf_abs_dev)
            )
            if val > alpha:
                conf_set.append(y)

        conf_set_list.append(conf_set)

    return conf_set_list


def extract_intervals(conf_set_list):
    # preallocate interval boundary matrix
    intervals = np.zeros((len(conf_set_list), 2))

    for i in range(len(conf_set_list)):
        intervals[i, 0] = np.min(conf_set_list[i])
        intervals[i, 1] = np.max(conf_set_list[i])

    return intervals


def pred_band_cdf_based(
    X_predict,
    X_train,
    y_train,
    X_conf,
    y_conf,
    alpha=0.1,
    y_grid=None,
    quantile_grid=None,
    nodesize=40,
):

    if quantile_grid is None:
        quantile_grid = np.linspace(0.001, 0.999, 100)

    if y_grid is None:
        y_vals_merged = np.concatenate((y_train.flatten(), y_conf.flatten()))
        y_grid_upper = np.max(y_vals_merged) + iqr(y_vals_merged)
        y_grid_lower = np.min(y_vals_merged) - iqr(y_vals_merged)
        y_grid = np.linspace(y_grid_lower, y_grid_upper, 15000)

    X_train_r = numpy_matrix_to_r_matrix(X_train)
    X_conf_r = numpy_matrix_to_r_matrix(X_conf)
    y_train_r = numpy_matrix_to_r_matrix(y_train)
    X_predict_r = numpy_matrix_to_r_matrix(X_predict)

    q_hat = np.array(
        quantreg_forest_grid(
            X_predict=X_predict_r,
            X_train=X_train_r,
            y_train=y_train_r,
            X_conf=X_conf_r,
            alpha_grid=quantile_grid,
            nodesize=nodesize,
        )
    )

    f_hat_conf, f_hat_y_mat = conditional_cdf_hat(
        y_grid=y_grid, y_vec=y_conf, q_hat_conf_mat=q_hat[0], q_hat_pred_mat=q_hat[1]
    )

    conf_set_list = p_y_func(
        alpha=alpha, y_grid=y_grid, f_hat_conf=f_hat_conf, f_hat_y_mat=f_hat_y_mat
    )

    res = extract_intervals(conf_set_list)

    return res


def flatten(l):
    new_l = []
    for tup in l:
        sublist = []
        for i, subelement in enumerate(tup):
            if isinstance(subelement, tuple):
                for j in subelement:
                    sublist.append(j)
            else:
                sublist.append(subelement)
        new_l.append(tuple(sublist))
    return new_l


def cond_variance(X_mat, error_type, linear_part=None):
    if error_type == "simple_linear":
        cond_variance = (X_mat.flatten()) ** 2

    elif error_type == "varying_squared_linear_part":
        cond_variance = 1 + linear_part ** 2
        # print(np.histogram(cond_variance))

    elif error_type == "varying_third_moment_mu":
        t_dist_part = 3.0 / (3 - 2)
        cond_variance = (
            t_dist_part
            * (1 + 2 * np.abs(linear_part) ** 3 / np.mean(np.abs(linear_part) ** 3))
            ** 2
        )

    else:
        raise ValueError("Please specify regular error_type.")

    return cond_variance


def x_scale(X_mat, error_type, linear_part=None):
    if error_type == "simple_linear":
        scale = X_mat.flatten()

    elif error_type == "varying_squared_linear_part":
        scale = linear_part

    elif error_type == "varying_third_moment_mu":
        scale = linear_part

    else:
        raise ValueError("Please specify regular error_type.")

    return scale


def construc_cond_metric_df(cond_variance, result_pred_bands, y_predict):

    interval_lengths = result_pred_bands[:, 1] - result_pred_bands[:, 0]
    covered = (y_predict.flatten() >= result_pred_bands[:, 0]) & (
        y_predict.flatten() <= result_pred_bands[:, 1]
    )
    # df = pd.DataFrame(np.stack((cond_variance, interval_lengths, covered), axis=1))
    df = np.stack((cond_variance, interval_lengths, covered), axis=1)
    return df


def construc_cond_metric_df_simulation(x_scale, result_pred_bands, y_predict):

    interval_lengths = result_pred_bands[:, 1] - result_pred_bands[:, 0]
    covered = (y_predict.flatten() >= result_pred_bands[:, 0]) & (
        y_predict.flatten() <= result_pred_bands[:, 1]
    )
    df = np.stack((x_scale, interval_lengths, covered), axis=1)
    return df


def generate_X(
    n,
    p,
    X_dist="normal",
    cor="none",
    standardize=False,
    rho=0.15,
    k=5,
    alpha=5,
    uniform_lower=0,
    uniform_upper=1,
):

    # Generate X matrix
    if X_dist == "normal":
        X = np.random.normal(0, 1, n * p).reshape((n, p))

    if X_dist == "binom":
        X = np.random.binomial(n=1, p=0.5, size=(n, p))

    if X_dist == "uniform":
        X = np.random.uniform(uniform_lower, uniform_upper, n * p).reshape((n, p))

    if X_dist == "skewed_normal":
        X = skewnorm.rvs(alpha, size=n * p).reshape((n, p))

    if X_dist == "mixture":
        X = np.zeros(n * p).reshape((n, p))

        x1 = np.random.normal(0, 1, n * p).reshape((n, p))
        x2 = np.random.binomial(n=1, p=0.5, size=(n, p))
        x3 = skewnorm.rvs(5, size=n * p).reshape((n, p))

        u = np.random.uniform(0, 1, p)
        i1 = u <= 1 / 3
        i2 = (1 / 3 < u) & (u <= 2 / 3)
        i3 = u > 2 / 3

        X[:, i1] = x1[:, i1]
        X[:, i2] = x2[:, i2]
        X[:, i3] = x3[:, i3]

    # Pairwise correlation
    if cor == "pair":
        b = (-2 * np.sqrt(1 - rho) + 2 * np.sqrt((1 - rho) + p * rho)) / (2 * p)
        a = b + np.sqrt(1 - rho)

        # calculate symmetric square root of p x p matrix whose diagonals are 1 and off diagonals are rho:
        sig_half = np.full(shape=(p, p), fill_value=b)
        np.fill_diagonal(sig_half, a)
        X = X @ sig_half

    # Auto-correlation
    if cor == "auto":
        for j in range(p):
            mat = X[:, max(0, j - k) : j + 1]
            wts = np.random.uniform(0, 1, mat.shape[1]).flatten()
            wts = wts / np.sum(wts)
            tmp = mat * wts
            X[:, j] = np.array(np.mean(tmp, axis=1))

    # Standardize, if necessary
    if standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    return X


def generate_y(
    X_mat,
    eps_dist="normal",
    error_type="const",
    functional_form="linear",
    sigma=1,
    force_beta_positive=True,
    non_zero_beta_count=None,
    magnitude_nonzero_coeffs=1,
    signal_noise_ratio=None,
    alpha=5,
    df=3,
):

    n, p = X_mat.shape

    if non_zero_beta_count is None:
        non_zero_beta_count = int(np.ceil(p / 10))

    if non_zero_beta_count is not None:
        if non_zero_beta_count > p:
            raise ValueError(
                "Number of non-zero coefficients cannot exceed the number of covariates in X."
            )
        else:
            non_zero_beta_count = int(non_zero_beta_count)

    # calculate the linear part of the conditional expectation function, or the error multiplicator:
    # Sample s variables uniformly at random, define true coefficients

    non_zero_coeffs = np.random.choice(p, size=non_zero_beta_count, replace=False)
    beta = np.zeros(p)
    beta[non_zero_coeffs] = np.random.choice(
        np.array([-1, 1]) * magnitude_nonzero_coeffs,
        size=non_zero_beta_count,
        replace=True,
    )
    if force_beta_positive:
        beta = np.abs(beta)
    linear_part = X_mat @ beta

    # main effect:
    if functional_form == "linear":
        mu = linear_part

    elif functional_form == "sine":
        mu = 2 * np.sin(np.pi * linear_part) + np.pi * linear_part

    elif functional_form == "stochastic_poisson":
        if p > 1:
            raise ValueError("This dgp can only be initialized with p = 1.")
        #    mu = np.zeros(n)
        #    for i in range(n):
        #        mu[i] = np.random.poisson(np.sin(X_mat.flatten()[i])**2 + 0.1)
        else:
            x = X_mat.flatten()
            ax = 0 * x
            for i in range(len(x)):
                ax[i] = np.random.poisson(np.sin(x[i]) ** 2 + 0.1) + 0.03 * x[
                    i
                ] * np.random.randn(1)
                ax[i] += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
            y = ax.astype(np.float32)
            return y

    else:
        raise ValueError("Please specify regular functional form.")

    # error:
    if eps_dist == "normal":
        eps = np.random.normal(0, 1, n)

    elif eps_dist == "uniform":
        eps = np.random.uniform(0, 1, n)

    elif eps_dist == "t":
        eps = np.random.standard_t(df=df, size=n)

    elif eps_dist == "skewed_normal":
        eps = skewnorm.rvs(alpha, size=n)

    # elif eps_dist=="mixed":
    #    eps_1 = np.random.normal(0, 1, n)
    #    tmp = np.random.uniform(0,1,n)
    #    tmp_2 = np.random.normal(0, 1, n)
    #    eps_2 = 25 * np.int64(tmp < 0.01) * tmp_2

    else:
        raise ValueError("Please specify regular error distribution.")

    if error_type == "const":
        sx = np.ones(n)
        sigma_vec = sigma * sx

    elif error_type == "simple_linear":
        sx = linear_part
        sigma_vec = sigma * sx

    elif error_type == "varying_third_moment_mu":
        sx = 1 + 2 * np.abs(linear_part) ** 3 / np.mean(np.abs(linear_part) ** 3)
        sigma_vec = sigma * sx

    elif error_type == "varying_squared_linear_part":
        sx = np.sqrt(1 + (linear_part) ** 2)
        sigma_vec = sigma * sx

    else:
        raise ValueError("Please specify regular error type.")

    assert eps.shape == (n,)
    assert sigma_vec.shape == (n,)
    assert mu.shape == (n,)

    if signal_noise_ratio is not None:
        mu = (
            mu
            * np.sqrt(signal_noise_ratio)
            * np.sqrt(np.mean(sigma_vec ** 2))
            / np.std(mu)
        )

    assert mu.shape == (n,)

    # if eps_dist=="mixed":
    #    y = mu + eps_1 * 0.03 * linear_part.flatten() + eps_2

    y = mu + eps * sigma_vec

    if functional_form != "stochastic_poisson":
        return y, eps, sigma_vec, mu, beta


def generate_X_fixed_positions(
    n,
    p,
    X_dist="normal",
    cor="none",
    standardize=False,
    rho=0.15,
    k=5,
    alpha=5,
    uniform_lower=0,
    uniform_upper=1,
):

    # Generate X matrix
    if X_dist == "normal":
        X = np.random.normal(0, 1, n * p).reshape((n, p))

    if X_dist == "binom":
        X = np.random.binomial(n=1, p=0.5, size=(n, p))

    if X_dist == "uniform":
        X = np.random.uniform(uniform_lower, uniform_upper, n * p).reshape((n, p))

    if X_dist == "skewed_normal":
        X = skewnorm.rvs(alpha, size=n * p).reshape((n, p))

    if X_dist == "mixture":
        X = np.zeros(n * p).reshape((n, p))

        x1 = np.random.normal(0, 1, n * p).reshape((n, p))
        x2 = np.random.binomial(n=1, p=0.5, size=(n, p))
        x3 = skewnorm.rvs(5, size=n * p).reshape((n, p))

        u = np.random.uniform(0, 1, p)
        i1 = u <= 1 / 3
        i2 = (1 / 3 < u) & (u <= 2 / 3)
        i3 = u > 2 / 3

        X[:, i1] = x1[:, i1]
        X[:, i2] = x2[:, i2]
        X[:, i3] = x3[:, i3]

        # setting the decisive 5 covariates to a fixed distribution for later purposes
        X[:, 0] = np.random.normal(0, 1, n)
        X[:, 4] = np.random.binomial(n=1, p=0.5, size=n)
        X[:, 6] = skewnorm.rvs(5, size=n)
        X[:, 8] = skewnorm.rvs(5, size=n)
        X[:, 9] = np.random.binomial(n=1, p=0.5, size=n)

    # Pairwise correlation
    if cor == "pair":
        b = (-2 * np.sqrt(1 - rho) + 2 * np.sqrt((1 - rho) + p * rho)) / (2 * p)
        a = b + np.sqrt(1 - rho)

        # calculate symmetric square root of p x p matrix whose diagonals are 1 and off diagonals are rho:
        sig_half = np.full(shape=(p, p), fill_value=b)
        np.fill_diagonal(sig_half, a)
        X = X @ sig_half

    # Auto-correlation
    if cor == "auto":
        for j in range(p):
            mat = X[:, max(0, j - k) : j + 1]
            wts = np.random.uniform(0, 1, mat.shape[1]).flatten()
            wts = wts / np.sum(wts)
            tmp = mat * wts
            X[:, j] = np.array(np.mean(tmp, axis=1))

    # Standardize, if necessary
    if standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    return X


def generate_y_fixed_positions_beta(
    X_mat,
    eps_dist="normal",
    error_type="const",
    functional_form="linear",
    sigma=1,
    force_beta_positive=True,
    non_zero_beta_count=None,
    magnitude_nonzero_coeffs=1,
    signal_noise_ratio=None,
    alpha=5,
    df=4,
):

    n, p = X_mat.shape

    if non_zero_beta_count is None:
        non_zero_beta_count = int(np.ceil(p / 10))

    if non_zero_beta_count is not None:
        if non_zero_beta_count > p:
            raise ValueError(
                "Number of non-zero coefficients cannot exceed the number of covariates in X."
            )
        else:
            non_zero_beta_count = int(non_zero_beta_count)

    # calculate the linear part of the conditional expectation function, or the error multiplicator:
    # Sample s variables uniformly at random, define true coefficients

    if eps_dist == "t":
        non_zero_coeffs = np.array([0, 4, 6, 8, 9])
        beta = np.zeros(p)
        beta[non_zero_coeffs] = np.random.choice(
            np.array([-1, 1]) * magnitude_nonzero_coeffs,
            size=non_zero_beta_count,
            replace=True,
        )
        if force_beta_positive:
            beta = np.abs(beta)
        linear_part = X_mat @ beta

    else:
        non_zero_coeffs = np.arange(non_zero_beta_count)
        beta = np.zeros(p)
        beta[non_zero_coeffs] = np.random.choice(
            np.array([-1, 1]) * magnitude_nonzero_coeffs,
            size=non_zero_beta_count,
            replace=True,
        )
        if force_beta_positive:
            beta = np.abs(beta)
        linear_part = X_mat @ beta

    # main effect:
    if functional_form == "linear":
        mu = linear_part

    elif functional_form == "sine":
        mu = 2 * np.sin(np.pi * linear_part) + np.pi * linear_part

    elif functional_form == "stochastic_poisson":
        if p > 1:
            raise ValueError("This dgp can only be initialized with p = 1.")
        #    mu = np.zeros(n)
        #    for i in range(n):
        #        mu[i] = np.random.poisson(np.sin(X_mat.flatten()[i])**2 + 0.1)
        else:
            x = X_mat.flatten()
            ax = 0 * x
            for i in range(len(x)):
                ax[i] = np.random.poisson(np.sin(x[i]) ** 2 + 0.1) + 0.03 * x[
                    i
                ] * np.random.randn(1)
                ax[i] += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
            y = ax.astype(np.float32)
            return y

    else:
        raise ValueError("Please specify regular functional form.")

    # error:
    if eps_dist == "normal":
        eps = np.random.normal(0, 1, n)

    elif eps_dist == "uniform":
        eps = np.random.uniform(0, 1, n)

    elif eps_dist == "t":
        eps = np.random.standard_t(df=df, size=n)

    elif eps_dist == "skewed_normal":
        eps = skewnorm.rvs(alpha, size=n)

    # elif eps_dist=="mixed":
    #    eps_1 = np.random.normal(0, 1, n)
    #    tmp = np.random.uniform(0,1,n)
    #    tmp_2 = np.random.normal(0, 1, n)
    #    eps_2 = 25 * np.int64(tmp < 0.01) * tmp_2

    else:
        raise ValueError("Please specify regular error distribution.")

    if error_type == "const":
        sx = np.ones(n)
        sigma_vec = sigma * sx

    elif error_type == "simple_linear":
        sx = linear_part
        sigma_vec = sigma * sx

    elif error_type == "varying_third_moment_mu":
        sx = 1 + 2 * np.abs(linear_part) ** 3 / 38.73
        sigma_vec = sigma * sx

    elif error_type == "varying_squared_linear_part":
        sx = np.sqrt(1 + (linear_part) ** 2)
        sigma_vec = sigma * sx

    else:
        raise ValueError("Please specify regular error type.")

    assert eps.shape == (n,)
    assert sigma_vec.shape == (n,)
    assert mu.shape == (n,)

    if signal_noise_ratio is not None:
        mu = (
            mu
            * np.sqrt(signal_noise_ratio)
            * np.sqrt(np.mean(sigma_vec ** 2))
            / np.std(mu)
        )

    assert mu.shape == (n,)

    # if eps_dist=="mixed":
    #    y = mu + eps_1 * 0.03 * linear_part.flatten() + eps_2

    y = mu + eps * sigma_vec

    if functional_form != "stochastic_poisson":
        return y, eps, sigma_vec, mu, beta


def get_conditional_variances(process_type):
    if (
        process_type == 3
    ):  # chernozhukov example distributional conformal prediction (2021)
        x_grid = np.linspace(0, 1, 2000)
        return x_grid, np.array(x_grid) ** 2
    if process_type == 4:  # romano table
        df = pd.read_csv("romano_table_cond_variances.csv")
        return np.array(df["X"]), np.array(df["cond_var"])
    if process_type == 2:
        x_grid = np.linspace(0, 5, 2000)
        cond_var = 1 + x_grid ** 2
        return x_grid, cond_var
    if process_type == 1:
        x_grid = np.linspace(-5, 5, 2000)
        cond_var = 2 * (1 + (2 * np.abs(x_grid) ** 3) / 38.73) ** 2
        return x_grid, cond_var


def get_cond_oracle_lenghts(process_type):
    if process_type == 1:
        df = pd.read_csv("lei_third_moment_table_oracle_lengths.csv")
        return np.array(df[:, 0]), np.array(df[:, 1])
    elif process_type == 2:
        df = pd.read_csv("candes_sine_table_oracle_lengths.csv")
        return np.array(df[:, 0]), np.array(df[:, 1])
    elif process_type == 3:
        df = pd.read_csv("chernozhukov_table_oracle_lengths.csv")
        return np.array(df[:, 0]), np.array(df[:, 1])
    elif process_type == 4:
        df = pd.read_csv("romano_table_oracle_lengths.csv")
        return np.array(df[:, 0]), np.array(df[:, 1])
    else:
        print("Specify valid process.")


def complex_treatment_effect_linear(n, p, treatment_to_noise_ratio, corr, signal_to_noise_ratio=1.0):
    # regressors:
    X = generate_X_fixed_positions(n = n, p=p, X_dist="normal", cor=corr, standardize=False, rho=0.5)
    # params_linear_means:
    beta = np.ones(p) * signal_to_noise_ratio
    #beta[::2] = signal_to_noise_ratio
    #beta[1::2] = -signal_to_noise_ratio
    #
    beta_treat = np.ones(p)
    half_point = round(p/2)
    beta_treat[:half_point] = treatment_to_noise_ratio
    beta_treat[half_point:] = -treatment_to_noise_ratio
    #beta_treat = np.ones(p) * treatment_to_noise_ratio
    # conditional means:
    mu_1 = X @ beta +  X @ beta_treat
    mu_0 = X @ beta
    # noise:
    eps_1 = np.random.normal(0, 1, n)
    eps_0 = np.random.normal(0, 1, n)
    # draw treatment assignment variable:
    W = np.random.binomial(n=1, p=0.5, size=(n,))
    # calculate other quantities of interest:
    ite = mu_1 - mu_0 + eps_1 - eps_0
    # observed y_obs depends on W:
    y_obs = W * (mu_1 + eps_1) + (1 - W) * (mu_0 + eps_0)
    return ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W

def constant_treatment_effect(n, p, treatment_to_noise_ratio, corr, signal_to_noise_ratio=1.0, sigma=1.0, scale_base_effect = True):
    # regressors:
    X = generate_X_fixed_positions(n = n, p=p, X_dist="normal", cor=corr, standardize=False, rho=0.5)
    # params_linear_means:
    beta = np.ones(p) * signal_to_noise_ratio
    #beta[::2] = signal_to_noise_ratio
    #beta[1::2] = -signal_to_noise_ratio
    # conditional means:
    mu_1 = X @ beta + treatment_to_noise_ratio
    mu_0 = X @ beta
    # noise:
    eps_1 = np.random.normal(0, 1, n)
    eps_0 = np.random.normal(0, 1, n)
    # draw treatment assignment variable:
    W = np.random.binomial(n=1, p=0.5, size=(n,))
    # calculate other quantities of interest:
    ite = mu_1 - mu_0 + eps_1 - eps_0
    # observed y_obs depends on W:
    y_obs = W * (mu_1 + eps_1) + (1 - W) * (mu_0 + eps_0)
    return ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W

def simple_treatment_effect(n, p, treatment_to_noise_ratio, corr, signal_to_noise_ratio=1.0):
    # regressors:
    X = generate_X_fixed_positions(n = n, p=p, X_dist="normal", cor=corr, standardize=False, rho=0.5)
    # params_linear_means:
    beta = np.ones(p) * signal_to_noise_ratio
    #beta[::2] = signal_to_noise_ratio
    #beta[1::2] = -signal_to_noise_ratio

    mu_0 = X @ beta + (X[:,0] > 0.5) * signal_to_noise_ratio
    mu_1 = mu_0 + treatment_to_noise_ratio * (X[:,1] > 0.1)
    # noise:
    eps_1 = np.random.normal(0, np.sqrt(2), n)
    eps_0 = np.random.normal(0, np.sqrt(2), n)
    # draw treatment assignment variable:
    W = np.random.binomial(n=1, p=0.5, size=(n,))
    # calculate other quantities of interest:
    ite = mu_1 - mu_0 + eps_1 - eps_0
    # observed y_obs depends on W:
    y_obs = W * (mu_1 + eps_1) + (1 - W) * (mu_0 + eps_0)
    return ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W

def complex_treatment_effect_nonlinear(n, p, treatment_to_noise_ratio, corr, signal_to_noise_ratio=1.0):
    # regressors:
    X = generate_X_fixed_positions(n = n, p=p, X_dist="normal", cor=corr, standardize=False, rho=0.5)
    # non-linear effect:
    factor = 12.0
    effect = (2 / (1 + np.exp(-factor*(X[:,0]-0.5)))) * (2 / (1 + np.exp(-factor*(X[:,1]-0.5))))
    mu_1 = signal_to_noise_ratio * effect + treatment_to_noise_ratio * effect 
    mu_0 = signal_to_noise_ratio * effect
    # noise:
    eps_1 = np.random.normal(0, 1, n)
    eps_0 = np.random.normal(0, 1, n)
    # draw treatment assignment variable:
    W = np.random.binomial(n=1, p=0.5, size=(n,))
    # calculate other quantities of interest:
    ite = mu_1 - mu_0 + eps_1 - eps_0
    # observed y_obs depends on W:
    y_obs = W * (mu_1 + eps_1) + (1 - W) * (mu_0 + eps_0)
    return ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W


def very_complex_treatment_effect(n, p, treatment_to_noise_ratio, corr, signal_to_noise_ratio=1.0):
    # regressors:
    X = generate_X_fixed_positions(n = n, p=p, X_dist="uniform", cor=corr, standardize=False, rho=0.5)
    # non-linear effect:
    beta = np.ones(p) * signal_to_noise_ratio
    #beta[::2] = 0.0

    #beta_treat = np.zeros(p)
    #beta_treat[0] = 1.0
    #beta_treat[5] = 1.0
    #beta_treat[9] = -1.0

    linear_part = X @ beta
    #linear_treat_part = X @ beta_treat
    #
    base_fun = 2 * np.sin(np.pi * linear_part)
    mu_1 = base_fun + (treatment_to_noise_ratio * ((X[:,1] > 0.1) - 0.5* (X[:,2] < 0.1) - 0.3 * (X[:,3] > 0.1)))
    #mu_1 = base_fun + treatment_to_noise_ratio * linear_treat_part
    mu_0 = base_fun
    # noise:
    eps_1 = np.random.normal(0, np.sqrt(2), n)
    eps_0 = np.random.normal(0, np.sqrt(2), n)
    # draw treatment assignment variable:
    W = np.random.binomial(n=1, p=0.5, size=(n,))
    # calculate other quantities of interest:
    ite = mu_1 - mu_0 + eps_1 - eps_0
    # observed y_obs depends on W:
    y_obs = W * (mu_1 + eps_1) + (1 - W) * (mu_0 + eps_0)
    return ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W
    


def get_oracle_interval(lower, upper):
    def oracle_interval(mean, var):
        std = np.sqrt(var)
        norm_obj = scipy.stats.norm(loc=mean,scale=std)
        quantiles = norm_obj.ppf([lower, upper])
        return quantiles
    return oracle_interval

def get_oracle_intervals(means, variances):
    oracle_interval_fun = get_oracle_interval(0.05, 0.95)
    result = list(map(oracle_interval_fun, means, variances))
    return result

def calc_normal_params(mu_1, mu_0, X, heteroscedastic):
    means = mu_1 - mu_0
    if heteroscedastic:
        variances =  X[:,0]**2 + np.ones(len(means))
    else:
        variances = np.ones(len(means)) * 2
    return means, variances


def share_signif_fun(oracle_intervals, ite_pred_intervals):
    which_oracle_ints_signif = np.logical_not((oracle_intervals[:,0] <= 0) & (oracle_intervals[:,1] >= 0))
    which_predicted_ints_signif = np.logical_not((ite_pred_intervals[:,0] <= 0) & (ite_pred_intervals[:,1] >= 0))
    oracle_signif_signs = np.sign(np.mean(oracle_intervals, axis=1))
    predicted_signif_signs = np.sign(np.mean(ite_pred_intervals, axis=1))
    same_sign = (oracle_signif_signs == predicted_signif_signs)
    correctly_signif_given_oracle_signif = which_oracle_ints_signif & which_predicted_ints_signif & same_sign
    if np.sum(which_oracle_ints_signif) == 0:
        return -1.0
    else:
        return np.sum(correctly_signif_given_oracle_signif) / np.sum(which_oracle_ints_signif)

def share_signif_oracles(oracle_intervals, ite_vals):
    which_oracle_ints_signif = np.logical_not((oracle_intervals[:,0] <= 0) & (oracle_intervals[:,1] >= 0))
    which_ites_not_zero = (ite_vals != 0)
    signif_oracles_given_ite_not_zero = which_oracle_ints_signif & which_ites_not_zero
    return np.sum(signif_oracles_given_ite_not_zero) / len(oracle_intervals)

def share_signif_intervals_given_ite_not_zero(ite_pred_intervals, ite_vals):
    which_predicted_ints_signif = np.logical_not((ite_pred_intervals[:,0] <= 0) & (ite_pred_intervals[:,1] >= 0))
    which_ites_not_zero = (ite_vals != 0)
    signif_intervals_given_ite_not_zero = which_predicted_ints_signif & which_ites_not_zero
    return np.sum(signif_intervals_given_ite_not_zero) / len(ite_pred_intervals)

def random_forest_regressor(X_predict, X_train, y_train, X_conf, params):
    
    model = RandomForestRegressor(**params)
    model.fit(X_train, y_train)

    mu_hat_x_conf = model.predict(X_conf)
    mu_hat_x_predict = model.predict(X_predict)
    mu_hat_x_train = model.predict(X_train)
    return mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf


def OLS_regressor(X_predict, X_train, y_train, X_conf, params):

    model = LinearRegression(**params)
    model.fit(X_train, y_train)

    mu_hat_x_conf = model.predict(X_conf)
    mu_hat_x_predict = model.predict(X_predict)
    mu_hat_x_train = model.predict(X_train)

    return mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf


def lgbm_regressor(X_predict, X_train, y_train, X_conf, params):
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)

    mu_hat_x_conf = model.predict(X_conf)
    mu_hat_x_predict = model.predict(X_predict)
    mu_hat_x_train = model.predict(X_train)

    return mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf


def truncate_keys(dictionary, name_estim):
    return dict((k.removeprefix(name_estim + "__"), v) for k, v in dictionary.items())

def extract_optimal_estimator_and_opt_params(df_cv_results, candidate_estimators, minimum_score_optimal):
    if minimum_score_optimal:
        df_opt = df_cv_results[df_cv_results["test_set_accuracy_best_params"] == df_cv_results["test_set_accuracy_best_params"].min()]
    else:
        df_opt = df_cv_results[df_cv_results["test_set_accuracy_best_params"] == df_cv_results["test_set_accuracy_best_params"].max()]

    df_opt_singleton = df_opt.sample()
    best_params_overall_pipe = df_opt_singleton["best_params"].item()
    # checks whih of the candidate estimators was actually used in the best pipe, and assigns the result to "name"
    for estim in candidate_estimators:
        for (k,v) in best_params_overall_pipe.items():
            if estim in k:
                name = estim
    # extracts from the optimal pipe only the params which relate to the estimator, and not preprocessing steps etc.
    opt_params_estimator_raw = {k: v for (k,v) in best_params_overall_pipe.items() if name in k} 
    # remove prefix from the optimal params:
    opt_params_estimator = truncate_keys(opt_params_estimator_raw, name)
    return name, opt_params_estimator


def pred_band_mean_based_new(
    X_predict,
    X_train,
    y_train,
    X_conf,
    y_conf,
    algorithm=random_forest_regressor,
    alpha=0.1,
    params=None
):

    # algorithm used to calculated mu_hat
    mu_hat_x_predict, mu_hat_x_train, mu_hat_x_conf = algorithm(
        X_predict=X_predict, X_train=X_train, y_train=y_train, X_conf=X_conf, params=params
    )

    conf_scores = absolute_dev_from_mean(
        y_conf=y_conf.flatten(), mu_hat_x_conf=mu_hat_x_conf.flatten()
    )
    k = (1 - alpha) * (1.0 / len(y_conf) + 1)
    d = np.quantile(conf_scores, k)

    pred_band_upper = mu_hat_x_predict + d
    pred_band_lower = mu_hat_x_predict - d

    res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)

    return res, mu_hat_x_predict



def dgp_const_treat(n, p, effect_size, main_effect_case="const"):

    X = generate_X_fixed_positions(n = n, p=p, X_dist="normal", cor="none", standardize=False, rho=0.5)

    if main_effect_case == "const":
        mu_1 = np.ones(n) + effect_size 
        mu_0 = np.ones(n)

    elif main_effect_case == "linear":
        beta = np.ones(p)
        beta[::2] = 0.0
        mu_1 = X @ beta + effect_size 
        mu_0 = X @ beta

    elif main_effect_case == "non-linear":
        beta = np.ones(p)
        beta[::2] = 0.0
        linear_part = X @ beta
        base_fun = 2 * np.log(1 + np.exp(linear_part))
        mu_1 = base_fun + effect_size
        mu_0 = base_fun

    else:
        raise ValueError("Please specify a valid main effect type.")

    # noise:
    eps_1 = np.random.normal(0, 1, n)
    eps_0 = np.random.normal(0, 1, n)
    # draw treatment assignment variable:
    W = np.random.binomial(n=1, p=0.5, size=(n,))
    # calculate other quantities of interest:
    ite = mu_1 - mu_0 + eps_1 - eps_0
    # observed y_obs depends on W:
    y_obs = W * (mu_1 + eps_1) + (1 - W) * (mu_0 + eps_0)
    return ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W


def generate_treatment_effects_helper(X, treatment_case):
    n, p = X.shape
    if treatment_case == "binary":
        condition = 1 * (X[:,0] > 0.0)
        treat = np.where(condition == 0, -1.0, condition)
        tau_x = treat

    elif treatment_case == "gaussian":
        beta_treat = np.ones(p)
        half_point = round(p/2)
        beta_treat[:half_point] = 1.0
        beta_treat[half_point:] = 0.0
        # division by true standard deviation of the sum to yield variance 1
        tau_x = (X @ beta_treat) / np.sqrt(half_point)

    else:
        raise ValueError("Please specify a valid main effect type.")

    return tau_x

def dgp_ate_zero(n, p, effect_size, main_effect_case="const", treatment_case="binary"):

    X = generate_X_fixed_positions(n = n, p=p, X_dist="normal", cor="none", standardize=False, rho=0.5)
    tau_x = generate_treatment_effects_helper(X=X, treatment_case=treatment_case)

    if main_effect_case == "const":
        mu_1 = np.ones(n) + effect_size * tau_x
        mu_0 = np.ones(n)

    elif main_effect_case == "linear":
        beta = np.ones(p)
        beta[::2] = 0.0
        mu_1 = X @ beta + effect_size * tau_x
        mu_0 = X @ beta

    elif main_effect_case == "non-linear":
        beta = np.ones(p)
        beta[::2] = 0.0
        linear_part = X @ beta
        base_fun = 2 * np.log(1 + np.exp(linear_part))
        mu_1 = base_fun + effect_size * tau_x
        mu_0 = base_fun

    else:
        raise ValueError("Please specify a valid main effect type.")

        
    # noise:
    eps_1 = np.random.normal(0, 1, n)
    eps_0 = np.random.normal(0, 1, n)
    # draw treatment assignment variable:
    W = np.random.binomial(n=1, p=0.5, size=(n,))
    # calculate other quantities of interest:
    ite = mu_1 - mu_0 + eps_1 - eps_0
    # observed y_obs depends on W:
    y_obs = W * (mu_1 + eps_1) + (1 - W) * (mu_0 + eps_0)
    return ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W