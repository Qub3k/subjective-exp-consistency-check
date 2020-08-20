# Authors:  Krzysztof Rusek <krusek@agh.edu.pl>
#           Jakub Nawała <jnawala@agh.edu.pl>

import numpy as np
import probability_grid_estimation as pge


def prob(psi, sigma, cdf=False):
    """

    :param psi: QNormal parameter, vector
    :param sigma: QNormal parameter, vector
    :param cdf: If true return pdf
    :return: probabilities
    """
    grid = pge.get_each_answer_probability_for_normal([psi], [sigma])
    probs = grid.to_numpy(dtype=np.float64)[0]
    if cdf:
        probs = np.cumsum(probs, axis=-1)
    return probs


def sample(psi, sigma, experiments, n):
    """

    :param psi: GSD parameter
    :param sigma: GSD parameter
    :param experiments: Number of testers
    :param n: number of samples
    :return: random sample from the QNormal distribution
    """

    probs = prob(psi, sigma)
    s = np.random.multinomial(experiments, probs, size=(n))
    return s
