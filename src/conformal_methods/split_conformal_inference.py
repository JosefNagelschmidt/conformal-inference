from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from utils import init_scoring_object, CQR_conformity_score

class SplitConformalRegressor(BaseEstimator, RegressorMixin):
    valid_methods_ = ["mean-based","weighted-mean-based","quantile-based","cdf-based"]
    
    def __init__(self, estimator = None, method= "mean-based", conf_size=0.5, param_grid=None):

        # self.regressor = regressor
        regressor = estimator()
        
        if isinstance(regressor, RegressorMixin):
            self.estimator = estimator
        else:
            raise ValueError("Invalid estimator argument. Must be a regressor.")
            
        if method in SplitConformalRegressor.valid_methods_:
            self.method = method
        else:
            raise ValueError("Invalid method. Allowed values are 'mean-based','weighted-mean-based','quantile-based', and 'cdf-based'.")
            
        if (conf_size > 0.0) & (conf_size < 1.0):
            self.conf_size = conf_size
        else:
            raise ValueError("Invalid size for the conformalization size. Accepted values are between 0 and 1.")
        
        self.param_grid = param_grid


    def _check_estimator(self, estimator = None):
        if estimator is None:
            raise ValueError("Please specify valid estimator.")

        regressor = estimator()
        if not hasattr(regressor, "fit") and not hasattr(regressor, "predict"):
            raise ValueError("Invalid estimator. Please provide a regressor with fit and predict methods.")
     
        return estimator
    
    def fit(self, X, y, params={}): 
        # checks
        estimator = self._check_estimator(self.estimator)
        X, y = check_X_y(X, y, force_all_finite=False, dtype=["float64", "int", "object"])
        # split sample
        X_train, X_conf, y_train, y_conf = train_test_split(X, y, test_size=self.conf_size)
        self.X_conf_ = X_conf
        self.y_conf_ = y_conf

        # training
        if self.method == "mean-based":
            self.mu_hat_ = estimator(**params).fit(X_train, y_train)
            
        if self.method == "weighted-mean-based":
            self.mu_hat_ = estimator(**params).fit(X_train, y_train)
            self.mu_hat_train_residuals_ = self.mu_hat_.predict(X_train)
            
            self.fitted_abs_res_ = np.abs(y_train.flatten() - self.mu_hat_train_residuals_.flatten())
            self.rho_hat_ = estimator(**params).fit(X_train, self.fitted_abs_res_)

        if self.method == "quantile-based":
            if ("quantile_to_fit" in params):
                if (len(params["quantile_to_fit"]) == 2):
                    # this quantile_to_fit param is unique for the quantregForest from Meinshausen (2006), 
                    # since alpha is the param for the quantiles to fit
                    self.fitted_quant_reg_forest_both_ = estimator(**params).fit(X_train, y_train)
                    self.quantiles_to_fit_ = np.array(params["quantile_to_fit"])
                else:
                    raise ValueError("The quantile_to_fit param must have length of two.")

            elif (isinstance(self.estimator(), BaseGradientBoosting)) & ("loss" in params):
                if (len(params["alpha"]) == 2):
                    lower_params = deepcopy(params)
                    upper_params = deepcopy(params)
                    lower_params["alpha"] = lower_params["alpha"][0]
                    upper_params["alpha"] = upper_params["alpha"][1]

                    self.fitted_quant_reg_forest_lower_ = estimator(**lower_params).fit(X_train, y_train)
                    self.fitted_quant_reg_forest_upper_ = estimator(**upper_params).fit(X_train, y_train)
                    self.quantiles_to_fit_ = np.array(params["alpha"])
                    
                else:
                    raise ValueError("The alpha param must have length of two.")

            else:
                raise ValueError("Invalid estimator or params. Please provide a quantile regression estimator with the corresponding params.")

        return self

    def tune(self, X, y, quantile=0.9, cv=5):
        estimator = self._check_estimator(self.estimator)
        X, y = check_X_y(X, y, force_all_finite=False, dtype=["float64", "int", "object"])
        
        if self.method == "quantile-based":
            self.param_grid["loss"] = ['quantile']
            self.param_grid["alpha"] = [quantile]
        if self.param_grid is not None:
            scoring_object = init_scoring_object(method=self.method, quantile=quantile)
            print(self.param_grid)
            #grid = GridSearchCV(self.regressor(**args), self.param_grid, cv=cv, scoring=scoring_object)
            grid = GridSearchCV(estimator(), self.param_grid, cv=cv, scoring=scoring_object)
            grid.fit(X, y)
            self.cv_results_ = grid.best_params_
            return grid
        else:
            raise ValueError("Invalid call, since no parameters for tuning have been defined.")

    def predict_intervals(self, X_pred, alpha=0.1):
        X_pred = check_array(X_pred, force_all_finite=False, dtype=["float64", "object"])
        
        if self.method == "mean-based":
            check_is_fitted(self,["mu_hat_"])
            y_pred_hat = self.mu_hat_.predict(X_pred)
            y_conf_hat = self.mu_hat_.predict(self.X_conf_)
            
            conf_scores = np.abs(self.y_conf_ - y_conf_hat)
            k = (1 - alpha) * (1.0 / len(self.y_conf_) + 1)
            d = np.quantile(conf_scores, k)

            pred_band_upper = y_pred_hat + d
            pred_band_lower = y_pred_hat - d

            res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)
            return res
        
        if self.method == "weighted-mean-based":
            check_is_fitted(self,["mu_hat_","mu_hat_train_residuals_","fitted_abs_res_","rho_hat_"])
            
            y_pred_hat = self.mu_hat_.predict(X_pred)
            y_conf_hat = self.mu_hat_.predict(self.X_conf_)
            
            y_pred_mad_hat = self.rho_hat_.predict(X_pred)
            y_conf_mad_hat = self.rho_hat_.predict(self.X_conf_)
            
            conf_scores = np.abs(self.y_conf_.flatten() - y_conf_hat.flatten()) / y_conf_mad_hat.flatten()
            k = (1 - alpha) * (1.0 / len(self.y_conf_) + 1)
            d = np.quantile(conf_scores, k)

            pred_band_upper = y_pred_hat + y_pred_mad_hat * d
            pred_band_lower = y_pred_hat - y_pred_mad_hat * d

            res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)
            return res
        
        if self.method == "quantile-based":
            if hasattr(self, "fitted_quant_reg_forest_both_"):
                check_is_fitted(self,["fitted_quant_reg_forest_both_"])
                self.quantiles_to_fit_
                assert len(self.quantiles_to_fit_) == 2
                # generate upper and lower quantile estimates at the same time for both sets
                y_conf_hat = self.fitted_quant_reg_forest_both_.predict(self.X_conf_)
                y_pred_hat = self.fitted_quant_reg_forest_both_.predict(X_pred)

                conf_scores = CQR_conformity_score(lower_quant_hat=y_conf_hat[:, 0], upper_quant_hat=y_conf_hat[:, 1], y_conf=self.y_conf_)
                k = (1 - alpha) * (1.0 / len(self.y_conf_) + 1)
                d = np.quantile(conf_scores, k)

                pred_band_upper = y_pred_hat[:, 1] + d
                pred_band_lower = y_pred_hat[:, 0] - d

                res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)

            elif (hasattr(self, "fitted_quant_reg_forest_lower_")) & (hasattr(self, "fitted_quant_reg_forest_upper_")):
                check_is_fitted(self,["fitted_quant_reg_forest_lower_"])
                check_is_fitted(self,["fitted_quant_reg_forest_upper_"])
                assert len(self.quantiles_to_fit_) == 2

                y_conf_hat_lower = self.fitted_quant_reg_forest_lower_.predict(self.X_conf_)
                y_conf_hat_upper = self.fitted_quant_reg_forest_upper_.predict(self.X_conf_)

                y_pred_hat_lower = self.fitted_quant_reg_forest_lower_.predict(X_pred)
                y_pred_hat_upper = self.fitted_quant_reg_forest_upper_.predict(X_pred)

                conf_scores = CQR_conformity_score(lower_quant_hat=y_conf_hat_lower, upper_quant_hat=y_conf_hat_upper, y_conf=self.y_conf_)
                k = (1 - alpha) * (1.0 / len(self.y_conf_) + 1)
                d = np.quantile(conf_scores, k)

                pred_band_upper = y_pred_hat_upper + d
                pred_band_lower = y_pred_hat_lower - d

                res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)
            
            else:
                raise ValueError("No correct quantile regressor was fitted previously.")

            return res
