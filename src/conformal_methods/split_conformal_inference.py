from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble._gb import BaseGradientBoosting
from copy import deepcopy
from scipy.stats import iqr


from src.conformal_methods.r_objects import QuantregForest
from src.conformal_methods.utils import init_scoring_object, CQR_conformity_score, conditional_cdf_hat, p_y_func, extract_intervals

class SplitConformalRegressor(BaseEstimator, RegressorMixin):
    valid_methods_ = ["mean-based","weighted-mean-based","quantile-based","cdf-based"]

    def __init__(self, estimator = None, method= "mean-based", conf_size=0.5, param_grid=None, quantiles_to_fit=np.array([0.05,0.95])):

        regressor = estimator()
        
        if isinstance(regressor, RegressorMixin):
            self.estimator = estimator
        else:
            raise ValueError("Invalid estimator argument. Must be a regressor.")
            
        if method in SplitConformalRegressor.valid_methods_:
            if (method == "quantile-based") & ((isinstance(regressor, QuantregForest)) | (isinstance(regressor, BaseGradientBoosting))):
                pass
            elif ((method == "mean-based") | (method == "weighted-mean-based")) & ~(isinstance(regressor, QuantregForest)):
                pass
            elif (method == "cdf-based") & (isinstance(regressor, QuantregForest)):
                pass
            else:
                raise ValueError("Invalid combination of input method and regressor.")
            self.method = method

        else:
            raise ValueError("Invalid method. Allowed values are 'mean-based','weighted-mean-based','quantile-based', and 'cdf-based'.")
            
        if (conf_size > 0.0) & (conf_size < 1.0):
            self.conf_size = conf_size
        else:
            raise ValueError("Invalid size for the conformalization size. Accepted values are between 0 and 1.")
        
        self.param_grid = param_grid
        self.quantiles_to_fit = quantiles_to_fit


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

        self.X_train_ = X_train
        self.y_train_ = y_train

        self.X_conf_ = X_conf
        self.y_conf_ = y_conf

        # training
        if self.method == "mean-based":
            if isinstance(params, dict):
                self.mu_hat_ = estimator(**params).fit(X_train, y_train)
            elif isinstance(params, list):
                self.mu_hat_ = estimator(**(params[0].best_params_)).fit(X_train, y_train)
            else:
                raise ValueError("Invalid type of parameter 'params'; must be dict or list type.")
            
        if self.method == "weighted-mean-based":
            if isinstance(params, dict):
                self.mu_hat_ = estimator(**params).fit(X_train, y_train)
                self.mu_hat_train_residuals_ = self.mu_hat_.predict(X_train)
                
                self.fitted_abs_res_ = np.abs(y_train.flatten() - self.mu_hat_train_residuals_.flatten())
                self.rho_hat_ = estimator(**params).fit(X_train, self.fitted_abs_res_)

            elif isinstance(params, list):
                self.mu_hat_ = estimator(**(params[0].best_params_)).fit(X_train, y_train)
                self.mu_hat_train_residuals_ = self.mu_hat_.predict(X_train)
                
                self.fitted_abs_res_ = np.abs(y_train.flatten() - self.mu_hat_train_residuals_.flatten())
                self.rho_hat_ = estimator(**(params[0].best_params_)).fit(X_train, self.fitted_abs_res_)
            else:
                raise ValueError("Invalid type of parameter 'params'; must be dict or list type.")

        if self.method == "quantile-based":
            # check if the tune method has been called before:
            # in that case, we have to modify the quantiles 
           
            if isinstance(estimator(), QuantregForest):
                # this is the untuned case when a simple dict is provided as params
                if isinstance(params, dict):
                    params["quantile_to_fit"] = self.quantiles_to_fit
                    if (len(params["quantile_to_fit"]) == 2):
                        # this quantile_to_fit param is unique for the quantregForest from Meinshausen (2006), 
                        # since alpha is the param for the quantiles to fit
                        self.fitted_quant_reg_forest_both_ = estimator(**params).fit(X_train, y_train)
                        self.quantiles_to_fit_ = params["quantile_to_fit"]
                    else:
                        raise ValueError("The quantile_to_fit param must have length of two.")
                # this is the tuned case, when a list of 2 optimal param cv_objects is provided        
                elif isinstance(params, list):
                    if (len(self.quantiles_to_fit) == 2) & (len(params) == 2):
                        params_lower = params[0].best_params_
                        params_lower["quantile_to_fit"] = self.quantiles_to_fit[0]
                        params_upper = params[1].best_params_
                        params_upper["quantile_to_fit"] = self.quantiles_to_fit[1]
                    
                        self.fitted_quant_reg_forest_lower_ = estimator(**params_lower).fit(X_train, y_train)
                        self.fitted_quant_reg_forest_upper_ = estimator(**params_upper).fit(X_train, y_train)
                        self.quantiles_to_fit_ = self.quantiles_to_fit
                    else:
                        raise ValueError("The 'quantiles_to_fit' param during initialization must have length of two.")

                else:
                    raise ValueError("Invalid type of parameter 'params'; must be dict or list type.")

            elif (isinstance(self.estimator(), BaseGradientBoosting)):
                assert len(self.quantiles_to_fit) == 2
                self.quantiles_to_fit_ = self.quantiles_to_fit
                if isinstance(params, dict):
                    params["loss"] = 'quantile'
                    params["alpha"] = self.quantiles_to_fit
                    lower_params = deepcopy(params)
                    upper_params = deepcopy(params)
                    lower_params["alpha"] = lower_params["alpha"][0]
                    upper_params["alpha"] = upper_params["alpha"][1]
                
                elif isinstance(params, list):
                    lower_params = params[0].best_params_
                    upper_params = params[1].best_params_

                    lower_params["loss"] = 'quantile'
                    lower_params["alpha"] = self.quantiles_to_fit[0]
                    upper_params["loss"] = 'quantile'
                    upper_params["alpha"] = self.quantiles_to_fit[1]

                else:
                    raise ValueError("Invalid type of parameter 'params'; must be dict or list type.")

                self.fitted_quant_reg_forest_lower_ = estimator(**lower_params).fit(X_train, y_train)
                self.fitted_quant_reg_forest_upper_ = estimator(**upper_params).fit(X_train, y_train)
                    
            else:
                raise ValueError("Invalid estimator or params. Please provide a quantile regression estimator with the corresponding params.")

        if self.method == "cdf-based":
            # valid params:
            # valid_params_cdf = ["nodesize", "mtry", "ntree"]

            if not isinstance(params, dict):
                raise ValueError("Only a dict of params is accepted, no tuning available at the moment.")

            if not isinstance(estimator(), QuantregForest):
                raise ValueError("Only a 'QuantregForest' from Meinshausen(2006) available at the moment.")

            y_grid_upper = np.max(y) + iqr(y)
            y_grid_lower = np.min(y) - iqr(y)

            self.y_grid_cdf_ = np.linspace(y_grid_lower, y_grid_upper, 15000)
            self.quantile_grid_cdf_ = np.linspace(0.001, 0.999, 100)
            if not "quantile_to_fit" in params:
                params["quantile_to_fit"] = self.quantile_grid_cdf_

            self.fitted_quant_reg_forest_cdf_ = estimator(**params).fit(X_train, y_train)
            
        return self

    def tune(self, X, y, quantile=np.array([0.9]), cv=5):
        # "quantile" param must be length 1 for mean based and weighted approach, and length 2 for quantile based approach;
        # also "quantile" must coincide with the "quantiles_to_fit" param during initialization of estimator for quantile based approach
        estimator = self._check_estimator(self.estimator)
        X, y = check_X_y(X, y, force_all_finite=False, dtype=["float64", "int", "object"])

        if self.method == "cdf-based":
            raise ValueError("This method cannot be tuned at the moment.")

        if not self.param_grid:
            raise ValueError("Please reinitialize the SplitConformalRegressor with a parameter grid for tuning.")
            
        # if self.method == "quantile-based":
        #     self.param_grid["loss"] = ['quantile']
        #     self.param_grid["alpha"] = [quantile]

        if self.param_grid is not None:
            if self.method != "quantile-based":
                assert len(quantile) == 1
                scoring_object = init_scoring_object(method=self.method, quantile=quantile)
                grid = GridSearchCV(estimator(), self.param_grid, cv=cv, scoring=scoring_object)
                grid.fit(X, y)
                # self.cv_results_ = grid.best_params_
                return [grid]
            else:
                #assert np.array_equal(quantile,self.quantiles_to_fit)
                assert len(self.quantiles_to_fit) == 2

                lower_params = deepcopy(self.param_grid)
                upper_params = deepcopy(self.param_grid)

                lower_params["loss"] = ['quantile']
                upper_params["loss"] = ['quantile']
                
                lower_params["alpha"] = [self.quantiles_to_fit[0]]
                upper_params["alpha"] = [self.quantiles_to_fit[1]]

                scoring_object_lower = init_scoring_object(method=self.method, quantile=self.quantiles_to_fit[0])
                grid_lower = GridSearchCV(estimator(), lower_params, cv=cv, scoring=scoring_object_lower)
                grid_lower.fit(X, y)

                scoring_object_upper = init_scoring_object(method=self.method, quantile=self.quantiles_to_fit[1])
                grid_upper = GridSearchCV(estimator(), upper_params, cv=cv, scoring=scoring_object_upper)
                grid_upper.fit(X, y)

                return [grid_lower, grid_upper]

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

        if self.method == "cdf-based":
            if hasattr(self, "fitted_quant_reg_forest_cdf_"):
                check_is_fitted(self,["fitted_quant_reg_forest_cdf_"])

                y_conf_hat = self.fitted_quant_reg_forest_cdf_.predict(self.X_conf_)
                y_pred_hat = self.fitted_quant_reg_forest_cdf_.predict(X_pred)
                
                f_hat_conf, f_hat_y_mat = conditional_cdf_hat(y_grid=self.y_grid_cdf_, 
                                                            y_vec=self.y_conf_, 
                                                            q_hat_conf_mat=y_conf_hat, 
                                                            q_hat_pred_mat=y_pred_hat)
                
                conf_set_list = p_y_func(alpha=alpha, y_grid=self.y_grid_cdf_, f_hat_conf=f_hat_conf, f_hat_y_mat=f_hat_y_mat)
                res = extract_intervals(conf_set_list)

            return res
    
    def predict(self, X_pred):
        X_pred = check_array(X_pred, force_all_finite=False, dtype=["float64", "object"])
        if self.method == "mean-based":
            check_is_fitted(self,["mu_hat_"])
            y_pred_hat = self.mu_hat_.predict(X_pred)
            return y_pred_hat

        else:
            raise ValueError("Not yet implemented.")