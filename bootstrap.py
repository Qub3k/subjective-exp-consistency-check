# Authors:  Krzysztof Rusek <krusek@agh.edu.pl>
#           Jakub Nawała <jnawala@agh.edu.pl>

import numpy as np
import gsd

def empirical_distribution(sample, cumulative=False):
    """
    Empirical (cumulative) distribution (ECD) function defined as `p(X<=x)`

    :param sample: array(num_samples, num_categories) containing counts
    :param cumulative: a flag indicating whether to return the empirical cumulative distribution function (ECDF) or
     an empirical probability mass function (EPMF)
    :return: EPMF or ECD if cumulative is True
    """
    normalizer = sample.sum(axis=-1)
    p_hat = sample / normalizer[:, np.newaxis]
    return np.cumsum(p_hat, axis=-1) if cumulative else p_hat


def l2_divergence(x, y):
    return np.sum((x - y) ** 2, axis=-1)


def test(sample_epdf, pdf, b_epdf, divergence=l2_divergence):
    """
    Example: Check unit test `bootstrap_test.py`
    :param sample_epdf: Epdf from observation (more precisely, empirical probability mass function)
    :param pdf: Theoretical pdf (actually, pmf) from estimated distribution
    :param b_epdf: Empirical pdfs of bootrapped samples from the estimated distribution
    :param divergence: Distance function e.g. L2 or chi-squared
    :return: Estimated probability of `divergence > divergence(sample_epdf, pdf)`
    """
    distances = divergence(b_epdf, pdf)
    observation_distance = divergence(sample_epdf, pdf)
    return np.mean(distances > observation_distance)


def T_statistic(n, p):
    """
    Calculates the T statistic (as defined by BĆ for the G-test)

    :param n: counts of observations in each cell (can be an array with dimensions num_samples x num_of_cells)
    :param p: expected probabilities of each cell (can be an array with dimensions num_samples x num_of_cells)
    :return: T statistic
    """
    n_total = np.sum(n, axis=-1, keepdims=True)
    T = n * np.log(n / (n_total * p))
    T = np.where(n == 0, 0, T).sum(axis=-1)
    return T


def G_test(n, p, m, q):
    """
    G-test,"Bogdan: stosujemy bootstrapową wersję (zamiast asymptotycznej ze względu na małe n) klasycznego testu o
    nazwie G-test czyli testu ilorazu wiarygodności."

    :param n: Observation counts `(n_1, n_2, n_3, n_4, n_5)`, a 1d array
    :param p: Estimated distribution `(p_1, p_2, p_3, p_4, p_5)`, a 1d array
    :param m: T Bootstrap samples from distribution `p`, array[T,5]
    :param q: T estimated distributions for bootstrapped samples, array[T,5]
    :return: G-test p-value
    """
    n_non_zero_cells = (n != 0).sum()
    if n_non_zero_cells == 1:
        return 1.0

    # Return a p-value of 1.0 only if exactly any two NEIGHBOURING cells are non-zero
    if n_non_zero_cells == 2:
        # Find indices of the top 2 elements
        top_two_idx = np.argpartition(n, -2)[-2:]
        idx_diff = np.abs(top_two_idx[0] - top_two_idx[1])
        # Only if the top 2 elements are neighbours, return 1.0
        if idx_diff == 1:
            return 1.0

    T = T_statistic(n, p)
    Tr = T_statistic(m, q)
    return np.mean(Tr >= T)
