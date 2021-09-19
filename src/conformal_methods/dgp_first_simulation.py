from scipy.stats import skewnorm
import numpy as np
from sklearn.preprocessing import StandardScaler


def generate_X(
    n,
    p,
    X_dist="normal",
    cor="none",
    standardize=False,
    rho=0.15,
    k=5,
    alpha=5,
    uniform_lower=0,
    uniform_upper=1,
    fixed_positions=True
):

    # Generate X matrix
    if X_dist == "normal":
        X = np.random.normal(0, 1, n * p).reshape((n, p))

    if X_dist == "binom":
        X = np.random.binomial(n=1, p=0.5, size=(n, p))

    if X_dist == "uniform":
        X = np.random.uniform(uniform_lower, uniform_upper, n * p).reshape((n, p))

    if X_dist == "skewed_normal":
        X = skewnorm.rvs(alpha, size=n * p).reshape((n, p))

    if X_dist == "mixture":

        X = np.zeros(n * p).reshape((n, p))

        x1 = np.random.normal(0, 1, n * p).reshape((n, p))
        x2 = np.random.binomial(n=1, p=0.5, size=(n, p))
        x3 = skewnorm.rvs(5, size=n * p).reshape((n, p))

        u = np.random.uniform(0, 1, p)
        i1 = u <= 1 / 3
        i2 = (1 / 3 < u) & (u <= 2 / 3)
        i3 = u > 2 / 3

        X[:, i1] = x1[:, i1]
        X[:, i2] = x2[:, i2]
        X[:, i3] = x3[:, i3]

        if fixed_positions:
            # setting the decisive 5 covariates to a fixed distribution for later purposes
            X[:, 0] = np.random.normal(0, 1, n)
            X[:, 4] = np.random.binomial(n=1, p=0.5, size=n)
            X[:, 6] = skewnorm.rvs(5, size=n)
            X[:, 8] = skewnorm.rvs(5, size=n)
            X[:, 9] = np.random.binomial(n=1, p=0.5, size=n)

    # Pairwise correlation
    if cor == "pair":
        b = (-2 * np.sqrt(1 - rho) + 2 * np.sqrt((1 - rho) + p * rho)) / (2 * p)
        a = b + np.sqrt(1 - rho)

        # calculate symmetric square root of p x p matrix whose diagonals are 1 and off diagonals are rho:
        sig_half = np.full(shape=(p, p), fill_value=b)
        np.fill_diagonal(sig_half, a)
        X = X @ sig_half

    # Auto-correlation
    if cor == "auto":
        for j in range(p):
            mat = X[:, max(0, j - k) : j + 1]
            wts = np.random.uniform(0, 1, mat.shape[1]).flatten()
            wts = wts / np.sum(wts)
            tmp = mat * wts
            X[:, j] = np.array(np.mean(tmp, axis=1))

    # Standardize, if necessary
    if standardize:
        scaler = StandardScaler().fit(X)
        X = scaler.transform(X)

    return X
