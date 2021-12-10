# Helper functions for working with the Simplified Li2020 model (sli for short).
#
# Author: Jakub Nawała <jakub.nawala@agh.edu.pl>

from scipy.stats import norm
import numpy as np


def prob(mos: np.ndarray, s_var: np.ndarray, cdf=False, precision=15):
    """
    Generates probabilities of observing each response category (5 in this case) for a set of Mean Opinion Scores
    (*mos*) and corresponding sample variances (*s_var*). If *cdf* is true, it returns values of the cumulative
    distribution function (CDF) for the five response categories.

    :param mos: vector of Mean Opinion Scores (MOS), that is, sample means
    :param s_var: vector of sample variances
    :param cdf: flag indicating whether to return probabilities of the five response categories (false, the default) or
     cumulative probabilities of these (true)
    :param precision: a number of decimal places to round to when checking whether the total probability truly equal 1.
     For example, if rounding to 3 decimal places, 0.9999 would produce 1.0, whereas 0.999 would not (it would produce
     0.999).
    :return: (no. of samples x no. of response categories) array with probabilities of each of the five
     response categories (1, 2, 3, 4 and 5) or with cumulative probabilities (if *cdf* is true)
    """
    # If sample variance equals 0, use the smallest available float as the scale of the normal distribution
    s_std_dev = np.sqrt(s_var)  # std_dev --- standard deviation
    s_std_dev = np.where(s_std_dev != 0, s_std_dev, np.finfo(np.float64).eps)
    # rv --- random variable
    rv = norm(loc=mos, scale=s_std_dev)
    p1 = rv.cdf(1.5)  # probability of observing responses with response category 1
    p2 = rv.cdf(2.5) - rv.cdf(1.5)  # probability of observing responses with response category 2
    p3 = rv.cdf(3.5) - rv.cdf(2.5)  # and so on...
    p4 = rv.cdf(4.5) - rv.cdf(3.5)
    p5 = 1 - rv.cdf(4.5)
    probs = np.stack([p1, p2, p3, p4, p5], axis=1)
    # Check whether the total probability equals 1
    total_prob = np.round(np.sum(probs, axis=-1), decimals=precision)
    assert all(np.equal(total_prob, 1)), f"Probabilities of all answers do not add up to 1 for at least one sample. " \
                                         f"Quitting. "
    if cdf:
        probs = np.cumsum(probs, axis=-1)
    # flatten the result if only one sample was processed
    if probs.shape[0] == 1:
        probs = probs.flatten()
    return probs


def sample(mos, s_var, n_subjects, n):
    """
    Generates *n* random samples with *n_subjects* observations each. The samples follow the Simplified Li2020 model
    defined for parameters *mos* and *s_var* (sample variance).

    :return: (n x 5) np.ndarray with frequencies of each of the five response categories
    """
    probs = prob(np.array([mos]), np.array([s_var]))
    s = np.random.multinomial(n_subjects, probs, size=(n))
    return s


def estimate_parameters(samples: np.ndarray, corrected=False):
    """
    Estimates Simplified Li2020 model's parameters.

    :param samples: a 2-dimensional (number of samples, number of response categories) np.ndarray with
     samples. Each sample represents the number of responses assigned to each response category.
    :param corrected: flag indicating whether to perform the corrected parameters estimation (True) or the standard
     one (False, the default)
    :return: a 2-dimensional np.ndarray with estimated MOSes (the first column) and sample variances (the second column)
     or None if something failed
    """
    n_subjects = np.sum(samples, axis=-1)
    mos_hat = np.sum(samples * [1, 2, 3, 4, 5], axis=-1) / n_subjects
    # Change response category frequencies into raw individual responses and compute variance for each sample
    # indvdl_resp --- individual responses
    indvdl_resp = np.apply_along_axis(lambda frequencies: np.repeat([1, 2, 3, 4, 5], frequencies), axis=1, arr=samples)
    var_hat = indvdl_resp.var(axis=-1)
    # Apply parameters estimation correction if this was requested. Done not to observe zero frequency cells.
    if corrected:
        std_dev_thresh = 1 / (2 * norm.ppf(1 - (1 / (2 * n_subjects))))
        std_dev_hat = np.sqrt(var_hat)
        var_hat = np.where(std_dev_hat < std_dev_thresh, std_dev_thresh ** 2, var_hat)
    mos_hat_var_hat = np.stack([mos_hat, var_hat], axis=1)
    return mos_hat_var_hat
