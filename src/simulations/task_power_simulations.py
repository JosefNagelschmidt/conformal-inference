import pytask

from src.config import BLD
from src.config import SRC

from src.conformal_methods.split_conformal_inference import SplitConformalRegressor
from src.conformal_methods.utils import get_oracle_intervals, calc_normal_params, dgp_ate_zero
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from itertools import product
import json
import numpy as np
import pandas as pd


def run_simulation(specs):
    
    if specs["treatment_case"][0] == "binary":
        treatment_to_noise_ratio = np.logspace(start=0.0, stop=0.9, num=specs["treatment_grid_density"])
    elif specs["treatment_case"][0] == "gaussian":
        treatment_to_noise_ratio = np.logspace(start=0.0, stop=1.23, num=specs["treatment_grid_density"])
    else:
        raise ValueError("Please specify correct treatment case param.")


    simulation_ids = np.arange(specs["n_sims"])

    index = product(specs["n"], specs["p"], treatment_to_noise_ratio, specs["main_effect_cases"], specs["treatment_case"], simulation_ids)

    index = pd.MultiIndex.from_tuples(index,
        names=("n", "p", "treatment_to_noise_ratio",
            "main_effect_case", "treatment_case", "simulation_id",))

    df = pd.DataFrame(
        columns=[
            "mean_oracle_length",
            "mse_train_y1",
            "opt_estimator",

            "mean_interval_length",
            "mean_coverage",

            "mean_interval_length_asy",
            "mean_coverage_asy",
            
            "mean_interval_length_naive",
            "mean_coverage_naive",
            
            "mean_interval_length_naive_asymp",
            "mean_coverage_naive_asymp",
        ],
        index=index,
    )


    same_case_as_previous_round = False

    param_grid_lin_reg = {"fit_intercept": [True]}
    param_grid_rf = {'n_estimators': [300], 'min_samples_split': [6, 15], "min_samples_leaf": [8]}
    param_grid_grad_boost = {'learning_rate': [0.1], 
                            'n_estimators': [300], 
                            'max_depth': [2],
                            "min_samples_leaf": [5]}

    for index in df.index:

        if index[5] != 0:
            same_case_as_previous_round = True
            for i in range(len(previous_index) - 1):
                if previous_index[i] != index[i]:
                    same_case_as_previous_round = False
        
        # generate train and test sample:
        ite, mu_1, mu_0, eps_1, eps_0, y_obs, X, W = dgp_ate_zero(n = index[0],
                                                                p = index[1], 
                                                                effect_size = index[2],
                                                                main_effect_case= index[3],
                                                                treatment_case = index[4])
    
        # draw new samples for later prediction:
        ite_pred, mu_1_pred, mu_0_pred, eps_1_pred, eps_0_pred, y_obs_pred, X_pred, W_pred = dgp_ate_zero(n = specs["pred_samples"],
                                                                        p = index[1], 
                                                                        effect_size = index[2],
                                                                        main_effect_case= index[3],
                                                                        treatment_case = index[4])

        lin_reg = SplitConformalRegressor(LinearRegression, method="mean-based", param_grid=param_grid_lin_reg, conf_size=0.3)
        rf_reg = SplitConformalRegressor(RandomForestRegressor, method="mean-based", param_grid=param_grid_rf, conf_size=0.3)
        grad_boost_reg = SplitConformalRegressor(GradientBoostingRegressor, method="mean-based", param_grid=param_grid_grad_boost, conf_size=0.3)

        # scalars that measure the best performance
        tuning_res_lin_reg = lin_reg.tune(X=X[W==1],y=y_obs[W==1], cv=2)
        tuning_res_rf_reg = rf_reg.tune(X=X[W==1],y=y_obs[W==1], cv=2)
        tuning_res_grad_boost_reg = grad_boost_reg.tune(X=X[W==1],y=y_obs[W==1], cv=2)

        models = {
            "lin_reg" : [lin_reg, tuning_res_lin_reg[0].best_score_, tuning_res_lin_reg],
            "rf_reg": [rf_reg, tuning_res_rf_reg[0].best_score_, tuning_res_rf_reg],
            "grad_boost_reg": [grad_boost_reg, tuning_res_grad_boost_reg[0].best_score_, tuning_res_grad_boost_reg]
        }

        # determine which model is closest to the true mean (MSE)
        optimal_regressor_score = tuning_res_lin_reg[0].best_score_
        optimal_regressor = lin_reg
        optimal_params = tuning_res_lin_reg
        optimal_key = "lin_reg"

        for key in models:
            if models[key][1] > optimal_regressor_score:
                optimal_regressor_score = models[key][1]
                optimal_regressor = models[key][0]
                optimal_params = models[key][2]
                optimal_key = key

    
        optimal_regressor = optimal_regressor.fit(X[W==1], y_obs[W==1], params=optimal_params)
        res_y_1_preds_0_05 = optimal_regressor.predict_intervals(X_pred=X_pred, alpha=0.05)
        res_y_1_preds_0_10 = optimal_regressor.predict_intervals(X_pred=X_pred, alpha=0.1)
        # for the conditional mean predictions:
        f_predict_1 = optimal_regressor.predict(X_pred=X_pred)
        # naive residuals:
        f_train_1 = optimal_regressor.predict(X_pred=optimal_regressor.X_train_)
        fitted_absolute_residuals_1 = np.abs(optimal_regressor.y_train_ - f_train_1)

        # refitting for the second counterfactual, one could tune again but this is likely to have a minor impact in our scenario
        optimal_regressor = optimal_regressor.fit(X[W==0], y_obs[W==0], params=optimal_params)
        res_y_0_preds_0_05 = optimal_regressor.predict_intervals(X_pred=X_pred, alpha=0.05)
        res_y_0_preds_0_10 = optimal_regressor.predict_intervals(X_pred=X_pred, alpha=0.1)
        # for the conditional mean predictions:
        f_predict_0 = optimal_regressor.predict(X_pred=X_pred)
        # naive residuals:
        f_train_0 = optimal_regressor.predict(X_pred=optimal_regressor.X_train_)
        fitted_absolute_residuals_0 = np.abs(optimal_regressor.y_train_ - f_train_0)
        
        # 1): determine simple method from paper with finite sample guarantee (conservative)
        
        lower = res_y_1_preds_0_05[:, 0] - res_y_0_preds_0_05[:, 1] 
        upper = res_y_1_preds_0_05[:, 1] - res_y_0_preds_0_05[:, 0] 
        ite_bands = np.stack((lower.flatten(), upper.flatten()), axis=1)                                                                                     
        length_bands = ite_bands[:, 1] - ite_bands[:, 0]
        mean_interval_length = np.mean(length_bands)
        in_the_range = np.sum((ite_pred.flatten() >= ite_bands[:, 0]) & (ite_pred.flatten() <= ite_bands[:, 1]))
        mean_coverage = in_the_range / len(ite_pred)
        
        # 2): determine asymptotic guarantee intervals from Kivaranovic, Ristl, et al. (2020):
        
        l_1_tilde = f_predict_1.flatten() - ((f_predict_1.flatten() - res_y_1_preds_0_10[:, 0].flatten()) / np.sqrt(2))
        l_0_tilde = f_predict_0.flatten() - ((f_predict_0.flatten() - res_y_0_preds_0_10[:, 0].flatten()) / np.sqrt(2))
        
        u_1_tilde = f_predict_1.flatten() + ((res_y_1_preds_0_10[:, 1].flatten() - f_predict_1.flatten()) / np.sqrt(2))
        u_0_tilde = f_predict_0.flatten() + ((res_y_0_preds_0_10[:, 1].flatten() - f_predict_0.flatten()) / np.sqrt(2))
        
        lower_asy = l_1_tilde - u_0_tilde
        upper_asy = u_1_tilde - l_0_tilde
        
        ite_bands_asy = np.stack((lower_asy.flatten(), upper_asy.flatten()), axis=1)
        length_bands_asy = ite_bands_asy[:, 1] - ite_bands_asy[:, 0]
        mean_interval_length_asy = np.mean(length_bands_asy)
        in_the_range_asy = np.sum(
            (ite_pred.flatten() >= ite_bands_asy[:, 0]) & (ite_pred.flatten() <= ite_bands_asy[:, 1])
        )
        mean_coverage_asy = in_the_range_asy / len(ite_pred)
        
        # 3): determine really naive pred bands, by using naive ite method and uncorrected residuals:
        
        d1 = np.quantile(fitted_absolute_residuals_1, 0.95)
        d0 = np.quantile(fitted_absolute_residuals_0, 0.95)

        pred_band_upper1_naive_pl = f_predict_1 + d1
        pred_band_lower1_naive_pl = f_predict_1 - d1
        
        pred_band_upper0_naive_pl = f_predict_0 + d0
        pred_band_lower0_naive_pl = f_predict_0 - d0
        
        res_y_1_naive_pl = np.stack((pred_band_lower1_naive_pl.flatten(), pred_band_upper1_naive_pl.flatten()), axis=1)
        res_y_0_naive_pl = np.stack((pred_band_lower0_naive_pl.flatten(), pred_band_upper0_naive_pl.flatten()), axis=1)
        
        lower_naive_pl = res_y_1_naive_pl[:, 0] - res_y_0_naive_pl[:, 1] 
        upper_naive_pl = res_y_1_naive_pl[:, 1] - res_y_0_naive_pl[:, 0] 
        
        ite_bands_naive_pl = np.stack((lower_naive_pl.flatten(), upper_naive_pl.flatten()), axis=1)
                                                                                                            
        length_bands_naive_pl = ite_bands_naive_pl[:, 1] - ite_bands_naive_pl[:, 0]
        mean_interval_length_naive_pl = np.mean(length_bands_naive_pl)

        in_the_range_naive_pl = np.sum(
            (ite_pred.flatten() >= ite_bands_naive_pl[:, 0]) & (ite_pred.flatten() <= ite_bands_naive_pl[:, 1])
        )
        mean_coverage_naive_pl = in_the_range_naive_pl / len(ite_pred)
        
        # 4): variation of asymptotic intervals from Kivaranovic, Ristl, et al. (2020), by using naive intervals in first step for counterfactuals 
        # (which do not have marginal coverage in finite, but asymptotic sample)
        d1_asymp = np.quantile(fitted_absolute_residuals_1, 0.9)
        d0_asymp = np.quantile(fitted_absolute_residuals_0, 0.9)
        
        pred_band_upper1_as = f_predict_1 + d1_asymp
        pred_band_lower1_as = f_predict_1 - d1_asymp
        
        pred_band_upper0_as = f_predict_0 + d0_asymp
        pred_band_lower0_as = f_predict_0 - d0_asymp
        
        res_y_1_as = np.stack((pred_band_lower1_as.flatten(), pred_band_upper1_as.flatten()), axis=1)
        res_y_0_as = np.stack((pred_band_lower0_as.flatten(), pred_band_upper0_as.flatten()), axis=1)
        
        l_1_tilde_as = f_predict_1.flatten() - ((f_predict_1.flatten() - res_y_1_as[:, 0].flatten()) / np.sqrt(2))
        l_0_tilde_as = f_predict_0.flatten() - ((f_predict_0.flatten() - res_y_0_as[:, 0].flatten()) / np.sqrt(2))
        
        u_1_tilde_as = f_predict_1.flatten() + ((res_y_1_as[:, 1].flatten() - f_predict_1.flatten()) / np.sqrt(2))
        u_0_tilde_as = f_predict_0.flatten() + ((res_y_0_as[:, 1].flatten() - f_predict_0.flatten()) / np.sqrt(2))
        
        lower_asy_naive = l_1_tilde_as - u_0_tilde_as
        upper_asy_naive = u_1_tilde_as - l_0_tilde_as
        
        ite_bands_asy_naive = np.stack((lower_asy_naive.flatten(), upper_asy_naive.flatten()), axis=1)
        
        length_bands_asy_naive = ite_bands_asy_naive[:, 1] - ite_bands_asy_naive[:, 0]
        mean_interval_length_asy_naive = np.mean(length_bands_asy_naive)
        in_the_range_asy_naive = np.sum(
            (ite_pred.flatten() >= ite_bands_asy_naive[:, 0]) & (ite_pred.flatten() <= ite_bands_asy_naive[:, 1])
        )
        mean_coverage_asy_naive = in_the_range_asy_naive / len(ite_pred)
        
        # at last, get oracle intervals for the ite's:
        oracle_ints = np.stack(get_oracle_intervals(*calc_normal_params(mu_1= mu_1_pred, mu_0= mu_0_pred, X =X_pred, heteroscedastic=False)), axis=0)
        length_oracle_ints = oracle_ints[:, 1] - oracle_ints[:, 0]
        mean_oracle_length = np.mean(length_oracle_ints)


        df.at[index, "mean_interval_length"] = mean_interval_length
        df.at[index, "mean_coverage"] = mean_coverage
        
        df.at[index, "mean_interval_length_asy"] = mean_interval_length_asy
        df.at[index, "mean_coverage_asy"] = mean_coverage_asy
        
        df.at[index, "mean_oracle_length"] = mean_oracle_length                                               
        df.at[index, "mse_train_y1"] = optimal_regressor_score                                                                                                                                                              
        df.at[index, "opt_estimator"] = optimal_key
        
        df.at[index, "mean_interval_length_naive"] = mean_interval_length_naive_pl
        df.at[index, "mean_coverage_naive"] = mean_coverage_naive_pl
       
        df.at[index, "mean_interval_length_naive_asymp"] = mean_interval_length_asy_naive
        df.at[index, "mean_coverage_naive_asymp"] = mean_coverage_asy_naive
    
        previous_index = index

    return df



@pytask.mark.parametrize("depends_on, produces",
    [
        (
            {
                "model": SRC / "simulations" / "specs" / f"{treatment_setup}.json",
                "script": SRC / "simulations" / "power_simulations.py"
            },
                BLD / "simulations" / "power_simulations" / f"df_results_{treatment_setup}.csv",
        )
        for treatment_setup in ["binary", "gaussian"]
    ],
)
def task_power_simulations(depends_on, produces):
    # dictionary imported into "specs":
    specs = json.loads(depends_on["model"].read_text(encoding="utf-8"))
    res_csv = run_simulation(specs)
    # Store resulting df for each specificiation:
    res_csv.to_csv(produces)