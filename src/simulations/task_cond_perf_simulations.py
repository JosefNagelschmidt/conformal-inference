from itertools import product
from sys import argv

import numpy as np
import pandas as pd
from functions import *
from statsmodels.nonparametric.kernel_regression import KernelReg


def conc_tuples(index_prep, method, i):
    conc = (method, i)
    return index_prep + conc


# (n, p, X_dist, X_correlation, eps_dist, error_type, functional_form, non_zero_beta_count, uniform_upper, standardized_X, method, sim_id)

n_sims = 5
process_type = int(argv[1])  # one of: 1,2,3,4

if process_type == 1:
    print("Initialize process 1.")
    index_prep = [
        # third moment error term:
        (
            200,
            10,
            "mixture",
            "pair",
            "t",
            "varying_third_moment_mu",
            "linear",
            5,
            1,
            True,
        )
    ]

elif process_type == 2:
    print("Initialize process 2.")
    index_prep = [
        # varying_squared_linear_part error term:
        (
            1000,
            100,
            "uniform",
            "none",
            "normal",
            "varying_squared_linear_part",
            "sine",
            5,
            1,
            False,
        )
    ]

elif process_type == 3:
    print("Initialize process 3.")
    index_prep = [
        # chernozhukov example distributional conformal prediction (2021)
        (1000, 1, "uniform", "none", "normal", "simple_linear", "linear", 1, 1, False)
    ]

elif process_type == 4:
    print("Initialize process 4.")
    index_prep = [
        # Conformalized Quantile Regression(2019), eq. 18
        (
            1000,
            1,
            "uniform",
            "none",
            "normal",
            "simple_linear",
            "stochastic_poisson",
            1,
            5,
            False,
        )
    ]

else:
    print("No process 1-4 specified.")

print(index_prep)

methods = {
    "pred_band_mean_based": pred_band_mean_based,
    "pred_band_weighted_mean_based": pred_band_weighted_mean_based,
    "pred_band_quantile_based_rf": pred_band_quantile_based_rf,
    "pred_band_cdf_based": pred_band_cdf_based,
}

methods_keys = list(methods.keys())

simulation_ids = np.arange(n_sims)

index = product(index_prep, methods_keys, simulation_ids)

index = flatten(l=list(index))

index = pd.MultiIndex.from_tuples(
    index,
    names=(
        "n",
        "p",
        "X_dist",
        "X_correlation",
        "eps_dist",
        "error_type",
        "functional_form",
        "non_zero_beta_count",
        "uniform_upper",
        "standardized_X",
        "method",
        "simulation_id",
    ),
)

df = pd.DataFrame(
    columns=[
        "mean_interval_length",
        "mean_coverage",
        "conditional_metrics_df",
        "nodesize",
    ],
    index=index,
)


same_case_as_previous_round = False

for index in df.index:

    if index[11] != 0:
        print("Previous index is: " + str(previous_index))
    print("Current index is: " + str(index))

    if index[11] != 0:
        same_case_as_previous_round = True
        for i in range(len(previous_index) - 1):
            if previous_index[i] != index[i]:
                same_case_as_previous_round = False

    # pred_samples = int(round(15000 / n_sims))
    pred_samples = 150
    total_sample = index[0] + pred_samples

    X = generate_X_fixed_positions(
        n=total_sample,
        p=index[1],
        X_dist=index[2],
        cor=index[3],
        standardize=index[9],
        uniform_upper=index[8],
    )

    if index[6] == "stochastic_poisson":
        y = generate_y_fixed_positions_beta(
            X_mat=X,
            eps_dist=index[4],
            error_type=index[5],
            functional_form=index[6],
            non_zero_beta_count=index[7],
        )

    else:
        y, eps, sigma_vec, mu, beta = generate_y_fixed_positions_beta(
            X_mat=X,
            eps_dist=index[4],
            error_type=index[5],
            functional_form=index[6],
            non_zero_beta_count=index[7],
        )

    X_predict, X_split_again, y_predict, y_split_again = train_test_split(
        X, y, train_size=pred_samples
    )
    X_train, X_conf, y_train, y_conf = train_test_split(
        X_split_again, y_split_again, test_size=0.5, train_size=0.5
    )

    if (index[11] == 0) or (not same_case_as_previous_round):
        if (index[10] == "pred_band_quantile_based_rf") or (
            index[10] == "pred_band_cdf_based"
        ):
            nodesize_opt, mat_overview = tune_nodesize_quantile_forest(
                X=X,
                y=y,
                nodesize_grid=[5,10,15,25,50],
                # nodesize_grid=[100],
                pred_band_method=methods[index[10]],
                n_simulations=1,
            )

        elif (index[10] == "pred_band_mean_based") or (
            index[10] == "pred_band_weighted_mean_based"
        ):
            nodesize_opt, mat_overview = tune_mean_based_rf(
                X=X,
                y=y,
                min_samples_leaf_grid=[5,10,15,25,50],
                # min_samples_leaf_grid=[100],
                pred_band_method=methods[index[10]],
                n_simulations=1,
            )

        else:
            raise ValueError("A problem with the prediction band method occured.")

    res = methods[index[10]](
        X_predict=X_predict,
        X_train=X_train,
        y_train=y_train,
        X_conf=X_conf,
        y_conf=y_conf,
        nodesize=int(nodesize_opt),
    )

    length_bands = res[:, 1] - res[:, 0]
    mean_interval_length = np.mean(length_bands)

    in_the_range = np.sum(
        (y_predict.flatten() >= res[:, 0]) & (y_predict.flatten() <= res[:, 1])
    )
    mean_coverage = in_the_range / len(y_predict)

    if index[5] == "simple_linear":  # these are process_types 3 and 4 (univariate)
        x_scale_diag = x_scale(X_mat=X_predict, error_type=index[5])

    else:
        linear_part = X_predict @ beta
        x_scale_diag = x_scale(
            X_mat=X_predict, error_type=index[5], linear_part=linear_part
        )

    cond_metrics_df = construc_cond_metric_df_simulation(
        x_scale=x_scale_diag, result_pred_bands=res, y_predict=y_predict
    )

    df.at[index, "mean_interval_length"] = mean_interval_length
    df.at[index, "mean_coverage"] = mean_coverage
    df.at[index, "conditional_metrics_df"] = cond_metrics_df
    df.at[index, "nodesize"] = nodesize_opt

    previous_index = index

    print("Nodesize opt is: " + str(nodesize_opt))

result = (
    df[["mean_interval_length", "mean_coverage"]].groupby(by=["method"]).sum() / n_sims
)
result.to_csv("process_" + str(process_type) + "_" + "x_scale_averages.csv")


for i in range(n_sims):
    if i == 0:
        res_mean_based = df.at[
            conc_tuples(index_prep=index_prep[0], method="pred_band_mean_based", i=i),
            "conditional_metrics_df",
        ]
        # res_mean_based = df.at[(200, 10, "mixture", "auto", "t", "varying_third_moment_mu", "linear", 5, 1, True, "pred_band_mean_based", i), "conditional_metrics_df"]
    else:
        tmp = df.at[
            conc_tuples(index_prep=index_prep[0], method="pred_band_mean_based", i=i),
            "conditional_metrics_df",
        ]
        res_mean_based = np.concatenate((res_mean_based, tmp), axis=0)

for i in range(n_sims):
    if i == 0:
        res_weighted_mean_based = df.at[
            conc_tuples(
                index_prep=index_prep[0], method="pred_band_weighted_mean_based", i=i
            ),
            "conditional_metrics_df",
        ]
    else:
        tmp = df.at[
            conc_tuples(
                index_prep=index_prep[0], method="pred_band_weighted_mean_based", i=i
            ),
            "conditional_metrics_df",
        ]
        res_weighted_mean_based = np.concatenate((res_weighted_mean_based, tmp), axis=0)

for i in range(n_sims):
    if i == 0:
        res_quantile_based = df.at[
            conc_tuples(
                index_prep=index_prep[0], method="pred_band_quantile_based_rf", i=i
            ),
            "conditional_metrics_df",
        ]
    else:
        tmp = df.at[
            conc_tuples(
                index_prep=index_prep[0], method="pred_band_quantile_based_rf", i=i
            ),
            "conditional_metrics_df",
        ]
        res_quantile_based = np.concatenate((res_quantile_based, tmp), axis=0)

for i in range(n_sims):
    if i == 0:
        res_cdf_based = df.at[
            conc_tuples(index_prep=index_prep[0], method="pred_band_cdf_based", i=i),
            "conditional_metrics_df",
        ]
    else:
        tmp = df.at[
            conc_tuples(index_prep=index_prep[0], method="pred_band_cdf_based", i=i),
            "conditional_metrics_df",
        ]
        res_cdf_based = np.concatenate((res_cdf_based, tmp), axis=0)


upper = 0.999
lower = 0.001


df_mean_based = pd.DataFrame(
    {
        "x_scale": res_mean_based[:, 0],
        "length": res_mean_based[:, 1],
        "coverage": res_mean_based[:, 2],
    }
)

df_w_mean_based = pd.DataFrame(
    {
        "x_scale": res_weighted_mean_based[:, 0],
        "length": res_weighted_mean_based[:, 1],
        "coverage": res_weighted_mean_based[:, 2],
    }
)

df_quantile_based = pd.DataFrame(
    {
        "x_scale": res_quantile_based[:, 0],
        "length": res_quantile_based[:, 1],
        "coverage": res_quantile_based[:, 2],
    }
)

df_cdf_based = pd.DataFrame(
    {
        "x_scale": res_cdf_based[:, 0],
        "length": res_cdf_based[:, 1],
        "coverage": res_cdf_based[:, 2],
    }
)


# if process_type == 1:
#    factor = 0.3
# else:
#    factor = 1

# Q1 = df_mean_based.quantile(lower)
# Q3 = df_mean_based.quantile(upper)
# IQR = Q3 - Q1

# df_mean_based_cleaned = df_mean_based[~((df_mean_based < (Q1 - 1.5 * IQR)) |(df_mean_based > (Q3 + 1.5 * IQR))).any(axis=1)]

Q3 = df_mean_based.x_scale.quantile(upper)
Q1 = df_mean_based.x_scale.quantile(lower)

df_mean_based_cleaned = df_mean_based[
    (df_mean_based.x_scale < Q3) & (df_mean_based.x_scale > Q1)
]

####

# Q1 = df_w_mean_based.quantile(lower)
# Q3 = df_w_mean_based.quantile(upper)
# IQR = Q3 - Q1

# df_w_mean_based_cleaned = df_w_mean_based[~((df_w_mean_based < (Q1 - 1.5 * IQR)) |(df_w_mean_based > (Q3 + 1.5 * IQR))).any(axis=1)]
Q3 = df_w_mean_based.x_scale.quantile(upper)
Q1 = df_w_mean_based.x_scale.quantile(lower)

df_w_mean_based_cleaned = df_w_mean_based[
    (df_w_mean_based.x_scale < Q3) & (df_w_mean_based.x_scale > Q1)
]


#####

# Q1 = df_quantile_based.quantile(lower)
# Q3 = df_quantile_based.quantile(upper)
# IQR = Q3 - Q1

# df_quantile_based_cleaned = df_quantile_based[~((df_quantile_based < (Q1 - 1.5 * IQR)) |(df_quantile_based > (Q3 + 1.5 * IQR))).any(axis=1)]

Q3 = df_quantile_based.x_scale.quantile(upper)
Q1 = df_quantile_based.x_scale.quantile(lower)

df_quantile_based_cleaned = df_quantile_based[
    (df_quantile_based.x_scale < Q3) & (df_quantile_based.x_scale > Q1)
]

#######

# Q1 = df_cdf_based.quantile(lower)
# Q3 = df_cdf_based.quantile(upper)
# IQR = Q3 - Q1

# df_cdf_based_cleaned = df_cdf_based[~((df_cdf_based < (Q1 - 1.5 * IQR)) |(df_cdf_based > (Q3 + 1.5 * IQR))).any(axis=1)]

Q3 = df_cdf_based.x_scale.quantile(upper)
Q1 = df_cdf_based.x_scale.quantile(lower)

df_cdf_based_cleaned = df_cdf_based[
    (df_cdf_based.x_scale < Q3) & (df_cdf_based.x_scale > Q1)
]


x_scales_merged = np.concatenate(
    (
        np.array(df_mean_based_cleaned["x_scale"]),
        np.array(df_w_mean_based_cleaned["x_scale"]),
        np.array(df_quantile_based_cleaned["x_scale"]),
        np.array(df_cdf_based_cleaned["x_scale"]),
    )
)


minimum = np.min(x_scales_merged)
maximum = np.max(x_scales_merged)

grid = np.linspace(minimum, maximum, 1000)

print("Start.")

df_mean_based_cleaned.to_csv("process_" + str(process_type) + "_" + "df_mean_based_cleaned.csv")
df_w_mean_based_cleaned.to_csv("process_" + str(process_type) + "_" + "df_w_mean_based_cleaned.csv")
df_quantile_based_cleaned.to_csv("process_" + str(process_type) + "_" + "df_quantile_based_cleaned.csv")
df_cdf_based_cleaned.to_csv("process_" + str(process_type) + "_" + "df_cdf_based_cleaned.csv")

for mode in ["coverage", "length"]:
    
    if mode == "coverage":

        print("Coverage stage.")

        kde_cov_mean_based = KernelReg(
            endog=df_mean_based_cleaned["coverage"],
            exog=df_mean_based_cleaned["x_scale"],
            var_type="o",
        )
        kernel_fit_cov_mean_based, marginal_cov_mean_based = kde_cov_mean_based.fit(
            data_predict=grid
        )
        ##
        print("Fitted mean based.")

        kde_cov_weighted_mean_based = KernelReg(
            endog=df_w_mean_based_cleaned["coverage"],
            exog=df_w_mean_based_cleaned["x_scale"],
            var_type="o",
        )
        (
            kernel_fit_cov_weigthed_mean_based,
            marginal_cov_weighted_mean_based,
        ) = kde_cov_weighted_mean_based.fit(data_predict=grid)
        ##
        print("Fitted w. mean based.")
        #
        kde_cov_quantile_based = KernelReg(
            endog=df_quantile_based_cleaned["coverage"],
            exog=df_quantile_based_cleaned["x_scale"],
            var_type="o",
        )
        (
            kernel_fit_cov_quantile_based,
            marginal_cov_quantile_based,
        ) = kde_cov_quantile_based.fit(data_predict=grid)
        ##
        print("Fitted quantile_based.")

        kde_cov_cdf_based = KernelReg(
            endog=df_cdf_based_cleaned["coverage"],
            exog=df_cdf_based_cleaned["x_scale"],
            var_type="o",
        )
        kernel_fit_cov_cdf_based, marginal_cov_cdf_based = kde_cov_cdf_based.fit(
            data_predict=grid
        )
        ###
        print("Fitted cdf_based.")

        dataset = pd.DataFrame(
            {
                "cond_variance_y_grid": grid,
                "mean_based_cond_coverage": kernel_fit_cov_mean_based,
                "w_mean_based_cond_coverage": kernel_fit_cov_weigthed_mean_based,
                "quantile_based_cond_coverage": kernel_fit_cov_quantile_based,
                "cdf_based_cond_coverage": kernel_fit_cov_cdf_based,
            }
        )

        dataset.to_csv(
            "process_" + str(process_type) + "_" + mode + "_x_scale" + ".csv"
        )

    elif mode == "length":

        print("Length stage.")

        kde_cov_mean_based = KernelReg(
            endog=df_mean_based_cleaned["length"],
            exog=df_mean_based_cleaned["x_scale"],
            var_type="c",
            reg_type="lc",
        )
        kernel_fit_cov_mean_based, marginal_cov_mean_based = kde_cov_mean_based.fit(
            data_predict=grid
        )
        ##
        print("Fitted mean based.")

        kde_cov_weighted_mean_based = KernelReg(
            endog=df_w_mean_based_cleaned["length"],
            exog=df_w_mean_based_cleaned["x_scale"],
            var_type="c",
        )
        (
            kernel_fit_cov_weigthed_mean_based,
            marginal_cov_weighted_mean_based,
        ) = kde_cov_weighted_mean_based.fit(data_predict=grid)
        ##
        print("Fitted w. mean based.")
        #
        kde_cov_quantile_based = KernelReg(
            endog=df_quantile_based_cleaned["length"],
            exog=df_quantile_based_cleaned["x_scale"],
            var_type="c",
        )
        (
            kernel_fit_cov_quantile_based,
            marginal_cov_quantile_based,
        ) = kde_cov_quantile_based.fit(data_predict=grid)
        ##
        print("Fitted quantile_based.")

        kde_cov_cdf_based = KernelReg(
            endog=df_cdf_based_cleaned["length"],
            exog=df_cdf_based_cleaned["x_scale"],
            var_type="c",
        )
        kernel_fit_cov_cdf_based, marginal_cov_cdf_based = kde_cov_cdf_based.fit(
            data_predict=grid
        )
        ###
        print("Fitted cdf_based.")

        dataset = pd.DataFrame(
            {
                "cond_variance_y_grid": grid,
                "mean_based_cond_length": kernel_fit_cov_mean_based,
                "w_mean_based_cond_length": kernel_fit_cov_weigthed_mean_based,
                "quantile_based_cond_length": kernel_fit_cov_quantile_based,
                "cdf_based_cond_length": kernel_fit_cov_cdf_based,
            }
        )

        dataset.to_csv(
            "process_" + str(process_type) + "_" + mode + "n_1000" + ".csv"
        )

    else:
        print("Mode not specified.")
