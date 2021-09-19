import numpy as np
import matplotlib.pyplot as plt

def testing_dgp_1(n):
    X = np.random.uniform(0,1,n).reshape((n,1))
    mu, sigma = 0, 1 # mean and standard deviation
    eps = np.random.normal(mu, sigma, n)
    y = X.flatten() + X.flatten() * eps.flatten()
    y = y.reshape(-1,1)
    return X, y.flatten()

def testing_dgp_2(n):
    X = np.random.normal(0, 1, n * 3).reshape((n, 3))
    eps = np.random.normal(0, 1, n)
    return X, eps.flatten()
