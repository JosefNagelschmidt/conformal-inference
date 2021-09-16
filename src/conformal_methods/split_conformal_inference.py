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
            # is it necessary to specify the alpha grid quantiles for later prediction in the params dictionary at this point
            self.fitted_quant_reg_forest_ = estimator(**params).fit(X_train, y_train)

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
            check_is_fitted(self,["fitted_quant_reg_forest_"])

            assert len(self.fitted_quant_reg_forest_.alpha) == 2
            y_conf_hat = self.fitted_quant_reg_forest_.predict(self.X_conf_)
            y_pred_hat = self.fitted_quant_reg_forest_.predict(X_pred)

            conf_scores = CQR_conformity_score(lower_quant_hat=y_conf_hat[:, 0], upper_quant_hat=y_conf_hat[:, 1], y_conf=self.y_conf_)
            k = (1 - alpha) * (1.0 / len(self.y_conf_) + 1)
            d = np.quantile(conf_scores, k)

            pred_band_upper = y_pred_hat[:, 1] + d
            pred_band_lower = y_pred_hat[:, 0] - d

            res = np.stack((pred_band_lower.flatten(), pred_band_upper.flatten()), axis=1)
            return res

