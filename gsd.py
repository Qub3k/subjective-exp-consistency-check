# Authors:  Krzysztof Rusek <krusek@agh.edu.pl>
#           Jakub Nawała <jnawala@agh.edu.pl>

# %%
import numpy as np
import probability_grid_estimation as pge


# %%
def prob(psi, rho, cdf=False):
    """
    :param psi: GSD parameter, vector
    :param rho: GSD parameter, vector
    :param cdf: If true returns the cumulative distribution function
    :return: probabilities of each answer
    """
    grid = pge.get_each_answer_probability_for_gsd([psi], [rho])
    probs = grid.to_numpy(dtype=np.float64)[0]
    if cdf:
        probs = np.cumsum(probs, axis=-1)
    return probs


def sample(psi, rho, experiments, n):
    """

    :param psi: GSD parameter
    :param rho: GSD parameter
    :param experiments: Number of testers
    :param n: number of samples
    :return: random sample from the GSD distribution
    """

    probs = prob(psi, rho)
    s = np.random.multinomial(experiments, probs, size=(n))
    return s
