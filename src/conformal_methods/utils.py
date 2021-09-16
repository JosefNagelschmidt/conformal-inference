import numpy as np 


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