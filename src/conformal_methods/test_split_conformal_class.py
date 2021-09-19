from _pytest.mark import param
import pytest
import numpy as np

from split_conformal_inference import SplitConformalRegressor

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from r_objects import QuantregForest
from sklearn.linear_model import ElasticNet

@pytest.fixture
def testing_dgp_10000(n=10000):
    X = np.random.normal(0, 1, n * 1).reshape((n, 1))
    eps = np.random.normal(0, 1, n)
    return X, eps.flatten()

@pytest.fixture
def testing_dgp_2000(n=2000):
    X = np.random.normal(0, 1, n * 1).reshape((n, 1))
    eps = np.random.normal(0, 1, n)
    return X, eps.flatten()

@pytest.fixture
def testing_dgp_quantiles_10000(n=1000):
    X = np.random.normal(0, 1, n * 5).reshape((n, 5))
    mu, sigma = 0, 1 # mean and standard deviation
    eps = np.random.normal(mu, sigma, n)
    y = ((X @ np.array([1.0, -4.0, 3.0, 0.3, 1.6]).reshape(-1,1))**2).flatten() + eps.flatten()
    assert y.shape == (n,)
    return X, y

@pytest.fixture
def testing_dgp_quantiles_2000(n=2000):
    X = np.random.normal(0, 1, n * 5).reshape((n, 5))
    mu, sigma = 0, 1 # mean and standard deviation
    eps = np.random.normal(mu, sigma, n)
    y = ((X @ np.array([1.0, -4.0, 3.0, 0.3, 1.6]).reshape(-1,1))**2).flatten() + eps.flatten()
    assert y.shape == (n,)
    return X, y


def pytest_namespace():
    return {'mean_length_baseline': 0.0}


def test_classifier_input():
    with pytest.raises(ValueError) as e_info:
        regressor = SplitConformalRegressor(GradientBoostingClassifier, method="mean-based")
    assert str(e_info.value) == "Invalid estimator argument. Must be a regressor."

def test_method_and_regressor_do_not_match():
    with pytest.raises(ValueError) as e_info:
        regressor = SplitConformalRegressor(QuantregForest, method="mean-based")
    assert str(e_info.value) == "Invalid combination of input method and regressor."

    with pytest.raises(ValueError) as e_info:
        regressor = SplitConformalRegressor(RandomForestRegressor, method="quantile-based")
    assert str(e_info.value) == "Invalid combination of input method and regressor."

    regressor = SplitConformalRegressor(QuantregForest, method="quantile-based")
    assert regressor.method == "quantile-based"

    regressor = SplitConformalRegressor(GradientBoostingRegressor, method="quantile-based")
    assert regressor.method == "quantile-based"

    regressor = SplitConformalRegressor(GradientBoostingRegressor, method="mean-based")
    assert regressor.method == "mean-based"


def test_param_grid_empty_for_tuning(testing_dgp_2000):
    X, y = testing_dgp_2000

    regressor = SplitConformalRegressor(GradientBoostingRegressor, method="quantile-based")
    with pytest.raises(ValueError) as e_info:
        res = regressor.tune(X=X,y=y)
    assert str(e_info.value) == "Please reinitialize the SplitConformalRegressor with a parameter grid for tuning."

    param_grid = {'nodesize': [2, 80]}
    regressor = SplitConformalRegressor(QuantregForest, method="quantile-based", param_grid=param_grid, quantiles_to_fit=np.array([0.05,0.95]))
    res = regressor.tune(X=X,y=y)


def test_mean_based_correct_fitting(testing_dgp_10000, testing_dgp_2000):
    X, y = testing_dgp_10000
    X_test, y_test = testing_dgp_2000

    regressor = SplitConformalRegressor(RandomForestRegressor, method="mean-based")
    pred_band = regressor.fit(X,y).predict_intervals(X_pred=X_test, alpha=0.1)
    # the mean of the prediction intervals should be approx zero, if fitted function is close to truth (which it should)
    np.testing.assert_almost_equal(np.mean(pred_band), 0.0, 1)

    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    length_non_tuned = pred_band[:,1] - pred_band[:,0]
    pytest.mean_length_baseline = np.mean(length_non_tuned)


def test_weighted_mean_based_correct_fitting(testing_dgp_10000, testing_dgp_2000):
    X, y = testing_dgp_10000
    X_test, y_test = testing_dgp_2000

    regressor = SplitConformalRegressor(RandomForestRegressor, method="weighted-mean-based")
    pred_band = regressor.fit(X,y).predict_intervals(X_pred=X_test, alpha=0.1)
    # the mean of the prediction intervals should be approx zero, if fitted function is close to truth (which it should)
    np.testing.assert_almost_equal(np.mean(pred_band), 0.0, 1)

    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

def test_quantile_based_correct_fitting(testing_dgp_10000, testing_dgp_2000):
    X, y = testing_dgp_10000
    X_test, y_test = testing_dgp_2000

    regressor = SplitConformalRegressor(GradientBoostingRegressor, method="quantile-based", quantiles_to_fit=np.array([0.05,0.95]))
    pred_band = regressor.fit(X,y).predict_intervals(X_pred=X_test, alpha=0.1)

    # the 0.05-quantile should be about -1.64 and the 0.95-quantile about 1.64 (oracle band)
    np.testing.assert_almost_equal(np.mean(pred_band[:,0]), -1.645, 1)
    np.testing.assert_almost_equal(np.mean(pred_band[:,1]), 1.645, 1)

    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    # second method for quantile based approach:
    regressor = SplitConformalRegressor(QuantregForest, method="quantile-based", quantiles_to_fit=np.array([0.05,0.95]), conf_size=0.25)
    pred_band = regressor.fit(X,y, params={"nodesize": 40}).predict_intervals(X_pred=X_test, alpha=0.1)

    # the following part is in the untuned case not guaranteed, and the random forests are often to conservative (therefore commented out)

    # the 0.05-quantile should be about -1.64 and the 0.95-quantile about 1.64 (oracle band)
    # np.testing.assert_almost_equal(np.mean(pred_band[:,0]), -1.645, 1)
    # np.testing.assert_almost_equal(np.mean(pred_band[:,1]), 1.645, 1)

    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    length_non_tuned = pred_band[:,1] - pred_band[:,0]
    mean_length_non_tuned = np.mean(length_non_tuned)

    assert pytest.mean_length_baseline != 0.0
    assert 0.9 * mean_length_non_tuned < pytest.mean_length_baseline 



def test_cdf_based_correct_fitting(testing_dgp_10000, testing_dgp_2000):
    X, y = testing_dgp_10000
    X_test, y_test = testing_dgp_2000

    with pytest.raises(ValueError) as e_info:
        regressor = SplitConformalRegressor(GradientBoostingRegressor, method="cdf-based", conf_size=0.35)
    assert str(e_info.value) == "Invalid combination of input method and regressor."

    regressor = SplitConformalRegressor(QuantregForest, method="cdf-based", conf_size=0.35)
    pred_band = regressor.fit(X, y, params={"nodesize": 40, "mtry": 1}).predict_intervals(X_pred=X_test, alpha=0.1)

    # the following part is in the untuned case not guaranteed, and the random forests are often to conservative (therefore commented out)

    # the 0.05-quantile should be about -1.64 and the 0.95-quantile about 1.64 (oracle band)
    # np.testing.assert_almost_equal(np.mean(pred_band[:,0]), -1.645, 1)
    # np.testing.assert_almost_equal(np.mean(pred_band[:,1]), 1.645, 1)

    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    length_non_tuned = pred_band[:,1] - pred_band[:,0]
    mean_length_non_tuned = np.mean(length_non_tuned)

    assert pytest.mean_length_baseline != 0.0
    assert 0.9 * mean_length_non_tuned < pytest.mean_length_baseline
    


def test_mean_based_correct_tuning(testing_dgp_10000, testing_dgp_2000):
    X, y = testing_dgp_10000
    X_test, y_test = testing_dgp_2000

    param_grid = {'n_estimators': [50, 100, 500], 'max_depth': [2,5,25], 'min_samples_leaf': [1, 3, 15]}
    regressor = SplitConformalRegressor(RandomForestRegressor, method="mean-based", param_grid=param_grid)
    tuning_res= regressor.tune(X=X,y=y, cv=2)
    opt_res = regressor.fit(X,y, params=tuning_res)
    pred_band = opt_res.predict_intervals(X_pred=X_test, alpha=0.1)
    # the mean of the prediction intervals should be approx zero, if fitted function is close to truth (which it should)
    np.testing.assert_almost_equal(np.mean(pred_band), 0.0, 1)

    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    # check if length is shorter than the non-tuned version:
    regressor_non_tuned = SplitConformalRegressor(RandomForestRegressor, method="mean-based")
    # strange behavior here, since regressor_non_tuned.fit(X,y, params={}) without the params keyword somehow gets passed 'loss': 'quantile' argument,
    # even though the default is params={}. Note: Seems to have to do with pytest, since in jupyter the code runs without error.
    pred_band_non_tuned = regressor_non_tuned.fit(X,y, params={}).predict_intervals(X_pred=X_test, alpha=0.1) 

    length_opt_band = pred_band[:,1] - pred_band[:,0]
    mean_length_opt = np.mean(length_opt_band)

    length_non_tuned = pred_band_non_tuned[:,1] - pred_band_non_tuned[:,0]
    mean_length_non_tuned = np.mean(length_non_tuned)

    assert mean_length_opt < mean_length_non_tuned

def test_weighted_mean_based_correct_tuning(testing_dgp_10000, testing_dgp_2000):
    X, y = testing_dgp_10000
    X_test, y_test = testing_dgp_2000

    param_grid = {'n_estimators': [50, 100, 500], 'max_depth': [2,5,25], 'min_samples_leaf': [1, 3, 15]}
    regressor = SplitConformalRegressor(RandomForestRegressor, method="weighted-mean-based", param_grid=param_grid)
    tuning_res= regressor.tune(X=X,y=y, cv=2)
    opt_res = regressor.fit(X,y, params=tuning_res)
    pred_band = opt_res.predict_intervals(X_pred=X_test, alpha=0.1)
    # the mean of the prediction intervals should be approx zero, if fitted function is close to truth (which it should)
    np.testing.assert_almost_equal(np.mean(pred_band), 0.0, 1)

    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    # check if length is shorter than the non-tuned version:
    regressor_non_tuned = SplitConformalRegressor(RandomForestRegressor, method="weighted-mean-based")
    # strange behavior here, since regressor_non_tuned.fit(X,y, params={}) without the params keyword somehow gets passed 'loss': 'quantile' argument,
    # even though the default is params={}. Note: Seems to have to do with pytest, since in jupyter the code runs without error.
    pred_band_non_tuned = regressor_non_tuned.fit(X,y, params={}).predict_intervals(X_pred=X_test, alpha=0.1) 

    length_opt_band = pred_band[:,1] - pred_band[:,0]
    mean_length_opt = np.mean(length_opt_band)

    length_non_tuned = pred_band_non_tuned[:,1] - pred_band_non_tuned[:,0]
    mean_length_non_tuned = np.mean(length_non_tuned)

    assert mean_length_opt < mean_length_non_tuned


def test_quantile_based_correct_tuning(testing_dgp_quantiles_10000, testing_dgp_quantiles_2000):
    X, y = testing_dgp_quantiles_10000
    X_test, y_test = testing_dgp_quantiles_2000

    param_grid = {'learning_rate': [0.01, 0.1, 0.5, 1.0], 'n_estimators': [100,200, 800], 'min_samples_leaf': [1, 5, 15, 35]}

    regressor = SplitConformalRegressor(GradientBoostingRegressor, method="quantile-based", param_grid=param_grid, quantiles_to_fit=np.array([0.05,0.95]), conf_size=0.25)
    tuning_res= regressor.tune(X=X,y=y, cv=3)
    opt_res = regressor.fit(X,y, params=tuning_res)
    pred_band = opt_res.predict_intervals(X_pred=X_test, alpha=0.1)
    
    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    # check if length is shorter than the non-tuned version:
    regressor_non_tuned = SplitConformalRegressor(GradientBoostingRegressor, method="quantile-based", quantiles_to_fit=np.array([0.05,0.95]), conf_size=0.25)
    # strange behavior here, since regressor_non_tuned.fit(X,y, params={}) without the params keyword somehow gets passed 'loss': 'quantile' argument,
    # even though the default is params={}. Note: Seems to have to do with pytest, since in jupyter the code runs without error.
    pred_band_non_tuned = regressor_non_tuned.fit(X,y, params={}).predict_intervals(X_pred=X_test, alpha=0.1) 

    length_opt_band = pred_band[:,1] - pred_band[:,0]
    mean_length_opt = np.mean(length_opt_band)

    length_non_tuned = pred_band_non_tuned[:,1] - pred_band_non_tuned[:,0]
    mean_length_non_tuned = np.mean(length_non_tuned)

    assert mean_length_opt < mean_length_non_tuned


def test_quantile_based_correct_tuning_QuantRegForest(testing_dgp_quantiles_10000, testing_dgp_quantiles_2000):
    X, y = testing_dgp_quantiles_10000
    X_test, y_test = testing_dgp_quantiles_2000

    param_grid = {'nodesize': [3, 5, 7, 15], 'mtry': [1,2,3]}

    regressor = SplitConformalRegressor(QuantregForest, method="quantile-based", param_grid=param_grid, quantiles_to_fit=np.array([0.05,0.95]), conf_size=0.4)
    tuning_res= regressor.tune(X=X,y=y, cv=2)
    opt_res = regressor.fit(X,y, params=tuning_res)
    pred_band = opt_res.predict_intervals(X_pred=X_test, alpha=0.1)
    
    in_the_range = np.sum((y_test.flatten() >= pred_band[:,0]) & (y_test.flatten() <= pred_band[:,1]))
    coverage = in_the_range / len(y_test)
    assert (0.86 < coverage) & (coverage < 0.94)

    # check if length is shorter than the non-tuned version:
    regressor_non_tuned = SplitConformalRegressor(QuantregForest, method="quantile-based", quantiles_to_fit=np.array([0.05,0.95]), conf_size=0.4)
    # strange behavior here, since regressor_non_tuned.fit(X,y, params={}) without the params keyword somehow gets passed 'loss': 'quantile' argument,
    # even though the default is params={}. Note: Seems to have to do with pytest, since in jupyter the code runs without error.
    pred_band_non_tuned = regressor_non_tuned.fit(X,y, params={}).predict_intervals(X_pred=X_test, alpha=0.1) 

    length_opt_band = pred_band[:,1] - pred_band[:,0]
    mean_length_opt = np.mean(length_opt_band)

    length_non_tuned = pred_band_non_tuned[:,1] - pred_band_non_tuned[:,0]
    mean_length_non_tuned = np.mean(length_non_tuned)

    assert mean_length_opt < mean_length_non_tuned

