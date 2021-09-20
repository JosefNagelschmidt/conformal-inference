import pytask

from src.config import BLD
from src.config import SRC

import numpy as np
import json
from itertools import product
import pandas as pd
from statsmodels.nonparametric.kernel_regression import KernelReg
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from src.conformal_methods.utils import flatten, generate_X_fixed_positions, generate_y_fixed_positions, x_scale, construc_cond_metric_df_simulation
from src.conformal_methods.split_conformal_inference import SplitConformalRegressor
from src.conformal_methods.r_objects import QuantregForest


def conc_tuples(index_prep, method, i):
    conc = (method, i)
    return index_prep + conc

def run_simulation(specs, produces):

    methods = ["mean-based", "weighted-mean-based", "quantile-based", "cdf-based"]
    simulation_ids = np.arange(specs["n_sims"])

    index_prep = [(
                specs["n"],
                specs["p"],
                specs["X_dist"],
                specs["X_correlation"],
                specs["eps_dist"],
                specs["error_type"],
                specs["functional_form"],
                specs["non_zero_beta_count"],
                specs["uniform_upper"],
                bool(int(specs["standardized_X"])),
            )
        ]

    index = product(index_prep, methods, simulation_ids)

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

       
        total_sample = index[0] + specs["pred_samples"]

        # draw samples:
        X = generate_X_fixed_positions(
            n=total_sample,
            p=index[1],
            X_dist=index[2],
            cor=index[3],
            standardize=index[9],
            uniform_upper=index[8],
        )

        if index[6] == "stochastic_poisson":
            y = generate_y_fixed_positions(
                X_mat=X,
                eps_dist=index[4],
                error_type=index[5],
                functional_form=index[6],
                non_zero_beta_count=index[7],
            )

        else:
            y, eps, sigma_vec, mu, beta = generate_y_fixed_positions(
                X_mat=X,
                eps_dist=index[4],
                error_type=index[5],
                functional_form=index[6],
                non_zero_beta_count=index[7],
            )

        X_predict, X_split_again, y_predict, y_split_again = train_test_split(
            X, y, train_size=specs["pred_samples"]
        )
        # X_train, X_conf, y_train, y_conf = train_test_split(
        #     X_split_again, y_split_again, train_size=specs["train_size"]
        # )

        if X_split_again.shape[1] > 1:
            max_features = round(X_split_again.shape[1] / 3)
        elif X_split_again.shape[1] == 1:
            max_features = 1
        else:
            raise ValueError("X has a dimensionality problem, missing regressors.")

        if (index[10] == "mean-based"):
            reg = SplitConformalRegressor(RandomForestRegressor, method="mean-based", conf_size=1-specs["train_size"], quantiles_to_fit=np.array([0.05,0.95]))
            reg = reg.fit(X=X_split_again, y=y_split_again, params={"min_samples_leaf": specs["nodesize"],
                                                              "max_features": max_features,
                                                              "n_estimators": specs["n_estimators"]})

            res = reg.predict_intervals(X_pred=X_predict, alpha=0.1)

        elif (index[10] == "weighted-mean-based"):
            reg = SplitConformalRegressor(RandomForestRegressor, method="weighted-mean-based", conf_size=1-specs["train_size"], quantiles_to_fit=np.array([0.05,0.95]))
            reg = reg.fit(X=X_split_again, y=y_split_again, params={"min_samples_leaf": specs["nodesize"],
                                                              "max_features": max_features,
                                                              "n_estimators": specs["n_estimators"]})

            res = reg.predict_intervals(X_pred=X_predict, alpha=0.1)
            
        elif (index[10] == "quantile-based"):
            reg = SplitConformalRegressor(QuantregForest, method="quantile-based", conf_size=1-specs["train_size"], quantiles_to_fit=np.array([0.05,0.95]))
            reg = reg.fit(X=X_split_again, y=y_split_again, params={"nodesize": specs["nodesize"], "mtry": max_features})
            res = reg.predict_intervals(X_pred=X_predict, alpha=0.1)

        elif (index[10] == "cdf-based"):
            reg = SplitConformalRegressor(QuantregForest, method="cdf-based", conf_size=1-specs["train_size"])
            reg = reg.fit(X=X_split_again, y=y_split_again, params={"nodesize": specs["nodesize"], "mtry": max_features})
            res = reg.predict_intervals(X_pred=X_predict, alpha=0.1)

        else:
            raise ValueError("Method misspecified.")

        # determine metrics:
        length_bands = res[:, 1] - res[:, 0]
        mean_interval_length = np.mean(length_bands)

        in_the_range = np.sum((y_predict.flatten() >= res[:, 0]) & (y_predict.flatten() <= res[:, 1]))
        mean_coverage = in_the_range / len(y_predict)

        # this determines which x-scale should be used for the later plots (X in univariate, or X*beta in multivariate case)
        if index[5] == "simple_linear":  # these are process_types 3 and 4 (univariate)
            x_scale_diag = x_scale(X_mat=X_predict, error_type=index[5])

        else:
            linear_part = X_predict @ beta
            x_scale_diag = x_scale(X_mat=X_predict, error_type=index[5], linear_part=linear_part)

        cond_metrics_df = construc_cond_metric_df_simulation(x_scale=x_scale_diag, result_pred_bands=res, y_predict=y_predict)

        df.at[index, "mean_interval_length"] = mean_interval_length
        df.at[index, "mean_coverage"] = mean_coverage
        df.at[index, "conditional_metrics_df"] = cond_metrics_df
        previous_index = index

    # after for loop and calculation, write average metrics into file:
    result = (df[["mean_interval_length", "mean_coverage"]].groupby(by=["method"]).sum() / specs["n_sims"])
    result.to_csv(produces["average_metrics_df"])

    # the following generates the kernel regression estimates for the four methods:
    for i in range(specs["n_sims"]):
        if i == 0:
            res_mean_based = df.at[conc_tuples(index_prep=index_prep[0], method="mean-based", i=i), "conditional_metrics_df",]
            # res_mean_based = df.at[(200, 10, "mixture", "auto", "t", "varying_third_moment_mu", "linear", 5, 1, True, "pred_band_mean_based", i), "conditional_metrics_df"]
        else:
            tmp = df.at[conc_tuples(index_prep=index_prep[0], method="mean-based", i=i), "conditional_metrics_df",]
            res_mean_based = np.concatenate((res_mean_based, tmp), axis=0)

    for i in range(specs["n_sims"]):
        if i == 0:
            res_weighted_mean_based = df.at[
                conc_tuples(
                    index_prep=index_prep[0], method="weighted-mean-based", i=i
                ),
                "conditional_metrics_df",
            ]
        else:
            tmp = df.at[
                conc_tuples(
                    index_prep=index_prep[0], method="weighted-mean-based", i=i
                ),
                "conditional_metrics_df",
            ]
            res_weighted_mean_based = np.concatenate((res_weighted_mean_based, tmp), axis=0)

    for i in range(specs["n_sims"]):
        if i == 0:
            res_quantile_based = df.at[
                conc_tuples(
                    index_prep=index_prep[0], method="quantile-based", i=i
                ),
                "conditional_metrics_df",
            ]
        else:
            tmp = df.at[
                conc_tuples(
                    index_prep=index_prep[0], method="quantile-based", i=i
                ),
                "conditional_metrics_df",
            ]
            res_quantile_based = np.concatenate((res_quantile_based, tmp), axis=0)

    for i in range(specs["n_sims"]):
        if i == 0:
            res_cdf_based = df.at[
                conc_tuples(index_prep=index_prep[0], method="cdf-based", i=i),
                "conditional_metrics_df",
            ]
        else:
            tmp = df.at[
                conc_tuples(index_prep=index_prep[0], method="cdf-based", i=i),
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

    Q3 = df_mean_based.x_scale.quantile(upper)
    Q1 = df_mean_based.x_scale.quantile(lower)

    df_mean_based_cleaned = df_mean_based[
        (df_mean_based.x_scale < Q3) & (df_mean_based.x_scale > Q1)
    ]

    Q3 = df_w_mean_based.x_scale.quantile(upper)
    Q1 = df_w_mean_based.x_scale.quantile(lower)

    df_w_mean_based_cleaned = df_w_mean_based[
        (df_w_mean_based.x_scale < Q3) & (df_w_mean_based.x_scale > Q1)
    ]

    Q3 = df_quantile_based.x_scale.quantile(upper)
    Q1 = df_quantile_based.x_scale.quantile(lower)

    df_quantile_based_cleaned = df_quantile_based[
        (df_quantile_based.x_scale < Q3) & (df_quantile_based.x_scale > Q1)
    ]

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
    df_mean_based_cleaned.to_csv(produces["conditional_res_mean_based"])
    df_w_mean_based_cleaned.to_csv(produces["conditional_res_w_mean_based"])
    df_quantile_based_cleaned.to_csv(produces["conditional_res_quantile_based"])
    df_cdf_based_cleaned.to_csv(produces["conditional_res_cdf_based"])

    # generate kernel estimates:
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

            dataset.to_csv(produces["final_kernel_estimated_coverage"])

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

            dataset.to_csv(produces["final_kernel_estimated_length"])

        else:
            print("Mode not specified.")


@pytask.mark.parametrize("depends_on, produces",
    [
        (
            {
                "type": SRC / "simulations" / "specs" / f"cond_sim_type_{type}.json",
            },
            {
                "average_metrics_df": BLD / "simulations" / "cond_perf_simulations" / f"average_results_{type}.csv",
                "conditional_res_mean_based": BLD / "simulations" / "cond_perf_simulations" / "cond_detailed_dfs" / f"cond_res_mean_based_{type}.csv",
                "conditional_res_w_mean_based": BLD / "simulations" / "cond_perf_simulations" / "cond_detailed_dfs" / f"cond_res_w_mean_based_{type}.csv",
                "conditional_res_quantile_based": BLD / "simulations" / "cond_perf_simulations" / "cond_detailed_dfs" / f"cond_res_quantile_based_{type}.csv",
                "conditional_res_cdf_based": BLD / "simulations" / "cond_perf_simulations" / "cond_detailed_dfs" / f"cond_res_cdf_based_{type}.csv",
                "final_kernel_estimated_coverage": BLD / "simulations" / "cond_perf_simulations" / f"kernel_coverage_results_{type}.csv",
                "final_kernel_estimated_length": BLD / "simulations" / "cond_perf_simulations" / f"kernel_length_results_{type}.csv",
            }
        )
        for type in [1,2,3,4]
    ],
)
def task_cond_perf_simulations(depends_on, produces):
    # dictionary imported into "specs":
    specs = json.loads(depends_on["type"].read_text(encoding="utf-8"))
    run_simulation(specs, produces)
