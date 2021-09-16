import rpy2.robjects as robjects
import rpy2.robjects.numpy2ri
from numba import jit
from rpy2.robjects.packages import importr

rpy2.robjects.numpy2ri.activate()
# import R's "base" package
base = importr("base")
# import R's "utils" package
utils = importr("utils")
quantregForest = importr("quantregForest")


from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

import numpy as np


def numpy_matrix_to_r_matrix(np_mat):
    if len(np_mat.shape) == 1:
        np_mat = np.atleast_2d(np_mat).T
    nr, nc = np_mat.shape
    r_mat = robjects.r.matrix(np_mat, nrow=nr, ncol=nc)
    return r_mat


class QuantregForest(BaseEstimator, RegressorMixin):
    def __init__(self, loss="quantile", alpha=np.array([0.9]), nodesize=5, mtry=1,
                 ntree=100): # params here are only specified due to GridSearchCV, which needs to find those as attributes
        self.loss = loss
        self.alpha = alpha
        self.nodesize = nodesize
        self.mtry = mtry
        self.ntree = ntree


    def fit(self, X, y):
        # check inputs
        X, y = check_X_y(X, y)

        X_r = numpy_matrix_to_r_matrix(X)
        y_r = numpy_matrix_to_r_matrix(y)

        quantreg_forest = robjects.r(
        """

            f_simple <- function(X_train, y_train, nodesize, mtry, ntree){
            qrf <- quantregForest(x=X_train, y=y_train, nodesize=nodesize, mtry=mtry, ntree=ntree)
            return(qrf)
            }
        """)

        self.fitted_quant_reg_forest_ = quantreg_forest(X_r, y_r, nodesize=self.nodesize, mtry=self.mtry, ntree=self.ntree)
        return self

    def predict(self, X):
        # checks:
        check_is_fitted(self)
        X = check_array(X)

        X_r = numpy_matrix_to_r_matrix(X)

        pred_forest = robjects.r(
        """
            f_pred <- function(forest, X_test, alpha){
            conditionalQuantile  <- predict(object=forest, newdata=X_test, what = alpha)
            return(conditionalQuantile)
            }
        """)
        
        preds = pred_forest(forest=self.fitted_quant_reg_forest_, X_test=X_r, alpha=self.alpha)
        return preds