import numpy as np 
from numba import jit
from scipy.stats import norm
from scipy.stats import skewnorm
from sklearn.preprocessing import StandardScaler


def init_scoring_object(method, quantile=0.9):
    def scoring_object(estimator, X, y):
        if (method == "mean-based") | (method == "weighted-mean-based"):
            #mu_hat = estimator.fit(X, y)
            #y_pred = mu_hat.predict(X)
            y_pred = estimator.predict(X)
            loss = np.mean((y - y_pred)**2)
            return -loss.item()
        if method == "quantile-based":
            #cond_quantile_hat = estimator.fit(X, y)
            y_pred = estimator.predict(X)
            if (quantile > 0) and (quantile < 1):
                residual = y - y_pred 
                return -np.sum(residual * (quantile - (residual<0)))
            else:
                return np.nan
    return scoring_object


def CQR_conformity_score(lower_quant_hat, upper_quant_hat, y_conf):
    first_arg = lower_quant_hat.flatten() - y_conf.flatten()
    second_arg = y_conf.flatten() - upper_quant_hat.flatten()
    conf_args = np.column_stack((first_arg, second_arg))
    return np.max(conf_args, axis=1)


def extract_intervals(conf_set_list):
    # preallocate interval boundary matrix
    intervals = np.zeros((len(conf_set_list), 2))

    for i in range(len(conf_set_list)):
        intervals[i, 0] = np.min(conf_set_list[i])
        intervals[i, 1] = np.max(conf_set_list[i])

    return intervals

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


def calc_normal_params(mu_1, mu_0, X, heteroscedastic):
    means = mu_1 - mu_0
    if heteroscedastic:
        variances =  X[:,0]**2 + np.ones(len(means))
    else:
        variances = np.ones(len(means)) * 2
    return means, variances


def get_oracle_interval(lower, upper):
    def oracle_interval(mean, var):
        std = np.sqrt(var)
        norm_obj = norm(loc=mean,scale=std)
        quantiles = norm_obj.ppf([lower, upper])
        return quantiles
    return oracle_interval

def get_oracle_intervals(means, variances):
    oracle_interval_fun = get_oracle_interval(0.05, 0.95)
    result = list(map(oracle_interval_fun, means, variances))
    return result


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

def generate_y_fixed_positions(
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
    y = mu + eps * sigma_vec

    if functional_form != "stochastic_poisson":
        return y, eps, sigma_vec, mu, beta


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

def construc_cond_metric_df_simulation(x_scale, result_pred_bands, y_predict):

    interval_lengths = result_pred_bands[:, 1] - result_pred_bands[:, 0]
    covered = (y_predict.flatten() >= result_pred_bands[:, 0]) & (
        y_predict.flatten() <= result_pred_bands[:, 1]
    )
    df = np.stack((x_scale, interval_lengths, covered), axis=1)
    return df
