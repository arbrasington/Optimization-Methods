import numpy as np
from scipy.stats import norm


def lower_confidence_bound(X, Y, model, kappa=1.96):
    mu, sigma = model.predict(X, return_std=True)
    return mu - kappa * sigma


def upper_confidence_bound(X, Y, model, kappa=1.96):
    mu, std = model.predict(X, return_std=True)
    return mu + kappa * std


def probability_improvement(X, Y, model, xi=0.01):
    mu, sigma = model.predict(X, return_std=True)
    values = np.zeros_like(mu)
    mask = sigma > 0
    mu_sample_opt = np.max(Y)
    improve = mu[mask] - mu_sample_opt - xi
    scaled = improve / sigma[mask]
    values[mask] = norm.cdf(scaled)

    return values


def expected_improvement(X, Y, gpr, xi=0.01):
    '''
    Computes the EI at points X based on existing samples X_sample
    and Y_sample using a Gaussian process surrogate model.

    Args:
        X: Points at which EI shall be computed (m x d).
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.
        xi: Exploitation-exploration trade-off parameter.

    Returns:
        Expected improvements at points X.
    '''
    mu, sigma = gpr.predict(X, return_std=True)
    mu_sample_opt = np.max(Y)

    values = np.zeros_like(mu)
    mask = sigma > 0
    improve = mu[mask] - mu_sample_opt - xi
    scaled = improve / sigma[mask]
    cdf = norm.cdf(scaled)
    pdf = norm.pdf(scaled)
    exploit = improve * cdf
    explore = sigma[mask] * pdf
    values[mask] = exploit + explore

    return values


from scipy.optimize import minimize


def propose_location(acquisition, X, Y, gpr, bounds, n_restarts=25):
    '''
    Proposes the next sampling point by optimizing the acquisition function.

    Args:
        acquisition: Acquisition function.
        X_sample: Sample locations (n x d).
        Y_sample: Sample values (n x 1).
        gpr: A GaussianProcessRegressor fitted to samples.

    Returns:
        Location of the acquisition function maximum.
    '''
    dim = X.shape[1]
    min_val = np.inf
    min_x = None

    def min_obj(X):
        # Minimization objective is the negative acquisition function
        return -acquisition(X.reshape(-1, dim), Y, gpr)

    # Find the best optimum by starting from n_restart different random points.
    for x0 in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, dim)):
        res = minimize(min_obj, x0=x0, bounds=bounds, method='L-BFGS-B')
        if res.fun < min_val:
            min_val = res.fun[0]
            min_x = res.x

    return min_x
