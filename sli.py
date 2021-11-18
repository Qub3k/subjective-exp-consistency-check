# Helper functions for working with the Simplified Li2020 model (sli for short).
#
# Author: Jakub Nawała <jakub.nawala@agh.edu.pl>

from scipy.stats import norm
import numpy as np


def prob(mos: np.ndarray, s_var: np.ndarray, cdf=False):
    """
    Generates probabilities of observing each response category (5 in this case) for a set of Mean Opinion Scores
    (*mos*) and corresponding sample variances (*s_var*). If *cdf* is true, it returns values of the cumulative
    distribution function (CDF) for the five response categories.

    :param mos: vector of Mean Opinion Scores (MOS), that is, sample means
    :param s_var: vector of sample variances
    :return: (no. of samples x no. of response categories) array with probabilities of each of the five
     response categories (1, 2, 3, 4 and 5) or with cumulative probabilities (if *cdf* is true)
    """
    # rv --- random variable
    rv = norm(loc=mos, scale=np.sqrt(s_var))
    p1 = rv.cdf(1.5)  # probability of observing responses with response category 1
    p2 = rv.cdf(2.5) - rv.cdf(1.5)  # probability of observing responses with response category 2
    p3 = rv.cdf(3.5) - rv.cdf(2.5)  # and so on...
    p4 = rv.cdf(4.5) - rv.cdf(3.5)
    p5 = 1 - rv.cdf(4.5)
    probs = np.stack([p1, p2, p3, p4, p5], axis=1)
    if cdf:
        probs = np.cumsum(probs, axis=-1)
    return probs


def sample(mos, s_var, n_subjects, n):
    """
    Generates *n* random samples with *n_subjects* observations each. The samples follow the Simplified Li2020 model
    defined for parameters *mos* and *s_var* (sample variance).

    :return: (n x 5) np.ndarray with frequencies of each of the five response categories
    """
    probs = prob(np.array([mos]), np.array([s_var]))
    s = np.random.multinomial(n_subjects, probs.flatten, size=(n))
    return s
