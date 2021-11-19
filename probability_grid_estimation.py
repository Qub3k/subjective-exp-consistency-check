# Calculates the probability of each discrete answer (i.e., 1, 2, 3, 4 or 5) assuming the normal model of subject
# responses. The normal model is understood as a normal distribution N(psi, sigma), where psi is a so called "true
# quality" and sigma is an error term.
#
# The probabilities are calculated for a grid of psi and sigma values.
#
# Author: Jakub Nawała <jnawala@agh.edu.pl>
# Date: Feb 25 2020

import logging
from _logger import setup_console_and_file_logger
import numpy as np
from scipy.stats import norm
from scipy.stats import chisquare
from scipy.stats import chi2
from itertools import product
import pandas as pd
from collections import Counter
import csv
from scipy.special import binom

logger = setup_console_and_file_logger(name=__name__, log_file_name="probability_grid_estimation.log",
                                       level=logging.INFO)

# Done to make sure this grid is computed only once
log_prob_grid_gsd_df = None
log_prob_grid_normal_df = None


def get_answer_counts(scores):
    """
    Returns a list with a count of each answer

    :param scores: a Pandas Series with scores for a given PVS
    :return: a list with a count of each answer
    """
    counts = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}
    counts.update(dict(Counter(scores).most_common()))
    counts_list = [count for score, count in counts.items()]
    return counts_list


def preprocess_real_data(subjective_datasets_csv_filepath="subjective_quality_datasets.csv",
                         should_also_group_by_exp=False, stimulus_identifier="PVS_id",
                         experiment_identifier="Experiment"):
    """
    Fetches the real-life subjective data and converts it to a format convenient for further processing.

    :param subjective_datasets_csv_filepath: a filepath to the CSV file with subjective data sets (the prepared for the
        original GSD paper)
    :param should_also_group_by_exp: a flag indicating whether to additionally group by experiment ID (this function
        groups only by PVS ID be default)
    :param stimulus_identifier: name of the column in the data that holds stimulus identifiers
    :param experiment_identifier: name of the column in the data that holds experiment identifiers. (This parameter is
        used only when *should_also_group_by_exp* is True.)
    :return: a DataFrameGroupBy object with the data grouped by the PVS_id column or by the PVS ID and Experiment
        columns (depending on the should_also_group_by_exp flag).
    """
    # If opening the one large tidy CSV file, read it properly
    if 'subjective_quality_datasets' in subjective_datasets_csv_filepath:
        logger.info("The one large tidy CSV file detected. Opening it properly...")
        subjective_datasets_df = pd.read_csv(subjective_datasets_csv_filepath, dtype={'PVS_id': str, 'SRC_id': str,
                                                                                      'HRC_id': str, 'Tester_id': str})
    else:
        subjective_datasets_df = pd.read_csv(subjective_datasets_csv_filepath)
    if should_also_group_by_exp:
        pvs_id_experiment_grouped = subjective_datasets_df.groupby([stimulus_identifier, experiment_identifier])
        return pvs_id_experiment_grouped
    else:
        pvs_id_grouped = subjective_datasets_df.groupby(stimulus_identifier)
        return pvs_id_grouped


def generate_sigma_grid():
    """
    Generates a grid of sigma values. Sigma values go linearly from 0.01 to 4.0 with the step of 0.01 and from there on
    with the step doubled each item, up to the sigma value of 40.

    :return: the grid of sigma values
    """
    sigma_grid = np.linspace(0.01, 4.0, num=401)
    current_sigma = 4.0
    step = 0.01
    while current_sigma <= 40.0:
        current_sigma += step
        sigma_grid = np.append(sigma_grid, current_sigma)
        step *= 2

    # Get rid of numbers like 4.000000000002
    sigma_grid = np.around(sigma_grid, decimals=2)

    return sigma_grid


def perform_chi_squared(sample, prob_grid_df, psi, sigma):
    """
    Performs the chi-squared goodness-of-fit test comparing observed counts of responses (from the sample) with those
    expected by the normal (or GSD) model (with estimated parameter values provided as psi and sigma function parameters
    ).

    :param sample: a sample of observations
    :param prob_grid_df: probabilities of observing each answer for each combination of psi and sigma/rho
    :param psi: a psi value estimated based on the sample
    :param sigma: a sigma/rho value estimated based on the sample
    :return: chi-squared statistic and p-value of the chi-squared goodness-of-fit test
    """
    n_observations = len(sample)

    obs_counts = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}
    obs_counts.update(dict(Counter(sample).most_common()))
    obs_counts_list = np.array([count for score, count in obs_counts.items()])

    exp_counts = prob_grid_df.loc[psi, sigma] * n_observations
    exp_counts_list = exp_counts.to_numpy().reshape(5)

    if all(exp_counts_list > 0):
        # 2 degrees of freedom because we have a 5-level scale (and thus, 4 degrees of freedom) and estimate two
        # parameters (and thus, we subtract 2 degrees of freedom). ddof stands for delta degrees of freedom. The actual
        # number of degrees of freedom is calculated as follows: k - 1 - ddof, were k is the number of observed
        # frequencies (five for the 5-level ACR scale).
        chi_squared, p_value = chisquare(obs_counts_list, exp_counts_list, ddof=2)
    else:
        # Calculate the chi-squared statistic and p-value by hand
        chi_squared = 0
        for obs_count, exp_count in zip(obs_counts_list, exp_counts_list):
            if exp_count == 0 and obs_count == 0:
                continue
            if exp_count == 0 and obs_count != 0:
                chi_squared = np.inf
                break
            chi_squared += (obs_count - exp_count) ** 2 / exp_count
        p_value = 1 - chi2.cdf(chi_squared, df=2)

    return chi_squared, p_value


def estimate_parameters(sample, prob_grid_df, sample_as_counts=False):
    """
    Estimate optimal parameters (psi and sigma/rho) of the QNormal (or GSD) model given the probability grid (for
    selected values of psi and sigma/rho) and a sample of observations.

    TODO Vectorize this function to support multiple samples provided with the "sample" parameter

    :param sample: a sample of discrete observations (expressed on the same scale as the one used to prepare the
        probability grid)
    :param prob_grid_df: a Data Frame with a probability grid of the QNormal (or GSD) model for selected values of psi
        and sigma/rho
    :param sample_as_counts: a flag indicating whether the input sample is expressed as counts of each score
    :return: estimated psi and sigma/rho parameter values (wrt. the probability grid provided) in the form of a tuple
    """
    if sample_as_counts:
        obs_counts_list = sample.tolist()
    else:
        obs_counts = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}
        obs_counts.update(dict(Counter(sample).most_common()))
        obs_counts_list = [count for score, count in obs_counts.items()]

    # Calculate the log-probability of providing each answer. Do it only once and reuse the existing grid in future
    # calls
    is_gsd = False
    is_normal = False
    if prob_grid_df.index.names[1] == "rho":
        is_gsd = True
    else:
        is_normal = True

    global log_prob_grid_gsd_df
    global log_prob_grid_normal_df
    if is_gsd:
        if log_prob_grid_gsd_df is None:
            logger.info("Calculating the log-probability grid for the GSD model")
            log_prob_grid_gsd_df = prob_grid_df.applymap(
                lambda prob: np.log(prob) if (prob != 0) else np.log(np.finfo(np.float64).eps))  # prob - probability
        log_prob_grid_df = log_prob_grid_gsd_df

    if is_normal:
        if log_prob_grid_normal_df is None:
            logger.info("Calculating the log-probability grid for the QNormal model")
            log_prob_grid_normal_df = prob_grid_df.applymap(
                lambda prob: np.log(prob) if (prob != 0) else np.log(np.finfo(np.float64).eps))
        log_prob_grid_df = log_prob_grid_normal_df

    # Calculate the log-likelihood for each psi and sigma/rho (_s means Pandas Series)
    log_prob_of_each_answer_df = log_prob_grid_df * obs_counts_list
    ll_grid_s = log_prob_of_each_answer_df.sum(axis='columns')  # sum along columns producing 1 value per row

    # Find index of the maximum log-likelihood, which corresponds to the estimated psi and sigma/rho
    estimated_psi_sigma_tuple = ll_grid_s.idxmax()
    return estimated_psi_sigma_tuple


def get_each_answer_probability_for_gsd(psi_grid, rho_grid, precision=15):
    """
    Calculates the probability of each of the five answers (i.e., 1, 2, 3, 4, and 5) for all combinations of psi and
    rho for the GSD model.

    :param psi_grid: an ndarray of psi values
    :param rho_grid: an ndarray of rho values
    :param precision: a number of decimal places to round to. For example, if rounding to 3 decimal places, 0.9999 would
        produce 1.0, whereas 0.999 would not (it would produce 0.999).
    :return: a Pandas Data Frame with a MultiIndex (psi and rho) and columns corresponding to each answer
        (i.e., 1, 2, 3, 4, and 5)
    """
    index = pd.MultiIndex.from_product([psi_grid, rho_grid], names=["psi", "rho"])
    prob_grid_df = pd.DataFrame(index=index, columns=[1, 2, 3, 4, 5])

    for psi_rho_tuple in product(*[psi_grid, rho_grid]):
        psi = psi_rho_tuple[0]
        rho = psi_rho_tuple[1]

        logger.debug("Processing psi of {} and rho of {}".format(psi, rho))
        if psi % 1.0 == 0.0 and rho == 0.0025:
            logger.info("Processing psi of {}".format(psi))

        vmax = (psi - 1) * (5 - psi)
        vmin = (np.ceil(psi) - psi) * (psi - np.floor(psi))
        if vmin == 0 and vmax == 0:
            c_psi = np.nan
        else:
            c_psi = 3 * vmax / (4 * (vmax - vmin))

        if rho < c_psi:
            rho_1_m_psi = rho * (1 - psi)
            denominator1 = 12 * c_psi - 8 * rho
            denominator2 = 8 * c_psi - 4 * rho
            denominator3 = 4 * c_psi
            prob_grid_df[1].loc[psi, rho] = (rho_1_m_psi / denominator1 + 1) * \
                                            (rho_1_m_psi / denominator2 + 1) * \
                                            (rho_1_m_psi / denominator3 + 1) * \
                                            ((1 - psi) / 4 + 1)
            prob_grid_df[2].loc[psi, rho] = 4 * (-rho_1_m_psi / denominator1) * \
                                            (rho_1_m_psi / denominator2 + 1) * \
                                            (rho_1_m_psi / denominator3 + 1) * \
                                            ((1 - psi) / 4 + 1)
            prob_grid_df[3].loc[psi, rho] = 6 * ((-rho_1_m_psi + 4 * c_psi - 4 * rho) / denominator1) * \
                                            (-rho_1_m_psi / denominator2) * \
                                            (rho_1_m_psi / denominator3 + 1) * \
                                            ((1 - psi) / 4 + 1)
            prob_grid_df[4].loc[psi, rho] = 4 * ((-rho_1_m_psi - 4 * c_psi) / denominator1 + 1) * \
                                            ((-rho_1_m_psi - 4 * c_psi) / denominator2 + 1) * \
                                            (-rho_1_m_psi / denominator3) * \
                                            ((1 - psi) / 4 + 1)
            prob_grid_df[5].loc[psi, rho] = ((-rho_1_m_psi - 4 * rho) / denominator1 + 1) * \
                                            ((-rho_1_m_psi - 4 * rho) / denominator2 + 1) * \
                                            ((-rho_1_m_psi - 4 * rho) / denominator3 + 1) * \
                                            ((psi - 1) / 4)
        elif psi == 5:
            prob_grid_df[1].loc[psi, rho], prob_grid_df[2].loc[psi, rho] = 0, 0
            prob_grid_df[3].loc[psi, rho], prob_grid_df[4].loc[psi, rho] = 0, 0
            prob_grid_df[5].loc[psi, rho] = 1

        elif psi == 1:
            prob_grid_df[1].loc[psi, rho], prob_grid_df[2].loc[psi, rho] = 1, 0
            prob_grid_df[3].loc[psi, rho], prob_grid_df[4].loc[psi, rho] = 0, 0
            prob_grid_df[5].loc[psi, rho] = 0
        else:
            for score in np.arange(1, 6):
                prob_grid_df[score].loc[psi, rho] = (rho - c_psi) / (1 - c_psi) * max(0, 1 - np.abs(score - psi)) + \
                                                    (1 - rho) / (1 - c_psi) * binom(4, score - 1) * \
                                                    (((psi - 1) / 4) ** (score - 1)) * (((5 - psi) / 4) ** (5 - score))

        total_prob = np.round(np.sum(prob_grid_df.loc[psi, rho]), decimals=precision)
        assert total_prob == 1, "Probabilities of all answers do not add up to 1. They add up to {} instead (when " \
                                "psi is {} and rho is {})".format(total_prob, psi, rho)

    return prob_grid_df


def get_each_answer_probability_for_qnormal(psi_grid, sigma_grid, precision=15):
    """
    Calculates the probability of each of the five answers (i.e., 1, 2, 3, 4, and 5) for all combinations of psi and
    sigma for the qnormal model.

    :param psi_grid: an ndarray of psi values
    :param sigma_grid: an ndarray of sigma values
    :param precision: a number of decimal places to round to. For example, if rounding to 3 decimal places, 0.9999 would
        produce 1.0, whereas 0.999 would not (it would produce 0.999).
    :return: a Pandas Data Frame with a MultiIndex (psi and sigma) and columns corresponding to each answer
        (i.e., 1, 2, 3, 4, and 5)
    """
    # Get rid of numbers like 4.000000000002
    psi_grid = np.around(psi_grid, decimals=2)
    sigma_grid = np.around(sigma_grid, decimals=2)

    index = pd.MultiIndex.from_product([psi_grid, sigma_grid], names=["psi", "sigma"])
    prob_grid_df = pd.DataFrame(index=index, columns=[1, 2, 3, 4, 5])

    for psi_sigma_tuple in product(*[psi_grid, sigma_grid]):
        psi = np.round(psi_sigma_tuple[0], decimals=2)
        sigma = np.round(psi_sigma_tuple[1], decimals=2)

        logger.debug("Processing psi of {} and sigma of {}".format(psi, sigma))
        if psi % 1.0 == 0.0 and sigma == 0.01:
            logger.info("Processing psi of {}".format(psi))

        # rv = random variable; o = a random variable describing subject responses (before discretisation and clipping)
        o_rv = norm(loc=psi, scale=sigma)
        prob_of_one = o_rv.cdf(1.5)
        prob_of_two = o_rv.cdf(2.5) - o_rv.cdf(1.5)
        prob_of_three = o_rv.cdf(3.5) - o_rv.cdf(2.5)
        prob_of_four = o_rv.cdf(4.5) - o_rv.cdf(3.5)
        prob_of_five = 1 - o_rv.cdf(4.5)

        total_prob = np.round(prob_of_one + prob_of_two + prob_of_three + prob_of_four + prob_of_five,
                              decimals=precision)

        assert total_prob == 1, "Probabilities of all answers do not add up to 1. They add up to {} instead (when " \
                                "psi is {} and sigma is {})".format(total_prob, psi, sigma)

        prob_grid_df[1].loc[psi, sigma] = prob_of_one
        prob_grid_df[2].loc[psi, sigma] = prob_of_two
        prob_grid_df[3].loc[psi, sigma] = prob_of_three
        prob_grid_df[4].loc[psi, sigma] = prob_of_four
        prob_grid_df[5].loc[psi, sigma] = prob_of_five

    return prob_grid_df


def perform_chi_squared_type2(sample, prob_grid_df, psi, sigma):
    """
    Performs the chi-squared goodness-of-fit test comparing observed counts of responses (from the sample) with those
    expected by the normal (or GSD) model (with estimated parameter values provided as psi and sigma function
    parameters). Importantly, this implementation removes from the test all cells with expected count below 1.

    :param sample: a sample of observations
    :param prob_grid_df: probabilities of observing each answer for each combination of psi and sigma/rho
    :param psi: a psi value estimated based on the sample
    :param sigma: a sigma/rho value estimated based on the sample
    :return: chi-squared statistic and p-value of the chi-squared goodness-of-fit test
    """
    n_observations = len(sample)

    obs_counts = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}
    obs_counts.update(dict(Counter(sample).most_common()))
    obs_counts_list = np.array([count for score, count in obs_counts.items()])

    exp_counts = prob_grid_df.loc[psi, sigma] * n_observations
    exp_counts_list = exp_counts.to_numpy().reshape(5)  # you may have to add [0] at the end

    if all(exp_counts_list > 1):
        # 2 degrees of freedom because we have a 5-level scale (and thus, 4 degrees of freedom) and estimate two
        # parameters (and thus, we subtract 2 degrees of freedom). ddof stands for delta degrees of freedom. The actual
        # number of degrees of freedom is calculated as follows: k - 1 - ddof, were k is the number of observed
        # frequencies (five for the 5-level ACR scale).
        chi_squared, p_value = chisquare(obs_counts_list, exp_counts_list, ddof=2)
    else:
        # Calculate the chi-squared statistic and p-value by hand
        chi_squared = 0
        # dof = degrees of freedom
        dof = len(obs_counts_list) - 1 - 2  # = 2 for the 5-level ACR scale and all cells retained
        for obs_count, exp_count in zip(obs_counts_list, exp_counts_list):
            if exp_count < 1 and obs_count == 0:
                dof = 1
                continue
            chi_squared += (obs_count - exp_count) ** 2 / exp_count
        p_value = 1 - chi2.cdf(chi_squared, df=dof)

    return chi_squared, p_value


def perform_chi_squared_type3(sample, prob_grid_df, psi, sigma):
    """
    Performs the chi-squared goodness-of-fit test comparing observed counts of responses (from the sample) with those
    expected by the normal (or GSD) model (with estimated parameter values provided as psi and sigma function
    parameters). Importantly, this implementation removes from the test all cells with observed counts being 0.

    :param sample: a sample of observations
    :param prob_grid_df: probabilities of observing each answer for each combination of psi and sigma/rho
    :param psi: a psi value estimated based on the sample
    :param sigma: a sigma/rho value estimated based on the sample
    :return: chi-squared statistic and p-value of the chi-squared goodness-of-fit test
    """
    n_observations = len(sample)

    obs_counts = {1.0: 0, 2.0: 0, 3.0: 0, 4.0: 0, 5.0: 0}
    obs_counts.update(dict(Counter(sample).most_common()))
    obs_counts_list = np.array([count for score, count in obs_counts.items()])

    exp_counts = prob_grid_df.loc[psi, sigma] * n_observations
    exp_counts_list = exp_counts.to_numpy().reshape(5)

    chi_squared = 0
    # dof = degrees of freedom = 2 for the 5-level ACR scale, all cells retained and 2 parameters estimated from the
    # sample
    dof = len(obs_counts_list) - 1 - 2
    for obs_count, exp_count in zip(obs_counts_list, exp_counts_list):
        if obs_count == 0:
            dof = 1
            continue
        chi_squared += (obs_count - exp_count) ** 2 / exp_count
    p_value = 1 - chi2.cdf(chi_squared, df=dof)

    return chi_squared, p_value


def main():
    # for quick testing use this: [1.1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 4.99]
    # psi_grid = np.linspace(1.01, 4.99, num=399)
    # psi_grid = np.around(psi_grid, decimals=2)  # gets rid of values like 4.0000003
    # sigma_grid = generate_sigma_grid()  # for quick testing use this: [0.01, 0.1, 0.3, 0.5, 0.7]
    # rho_grid = np.linspace(0.0025, 1.0, 400)  # for quick testing use this: [0.01, 0.1, 0.3, 0.5, 0.7, 0.99]
    # rho_grid = np.around(rho_grid, decimals=4)
    #
    # prob_grid_normal_df = get_each_answer_probability_for_normal(psi_grid, sigma_grid)
    # prob_grid_gsd_df = get_each_answer_probability_for_gsd(psi_grid, rho_grid)
    #
    # pickle_file_filename_normal = "normal_prob_grid.pkl"
    # pickle_file_filename_gsd = "gsd_prob_grid.pkl"
    # prob_grid_normal_df.to_pickle(pickle_file_filename_normal)
    # prob_grid_gsd_df.to_pickle(pickle_file_filename_gsd)
    # logger.info("The probability grid for the normal model stored in the {} file".format(pickle_file_filename_normal))
    # logger.info("The probability grid for the GSD model stored in the {} file".format(pickle_file_filename_gsd))

    # prob_grid_normal_df = pd.read_pickle("normal_prob_grid.pkl")
    prob_grid_gsd_df = pd.read_pickle("gsd_prob_grid.pkl")

    # Get the real life subjective scores grouped by PVS ID
    pvs_id_exp_grouped_scores = preprocess_real_data("opticom_res_tidy.csv", should_also_group_by_exp=True)

    csv_results_filename = "opticom_res_gsd_chi2_gof.csv"
    logger.info("Storing the results in the {} file".format(csv_results_filename))
    with open(csv_results_filename, 'w', newline='') as csvfile:
        fieldnames = ["PVS_id", "count1", "count2", "count3", "count4", "count5", "MOS", "Exp", "psi_hat", "rho_hat",
                      "chi2", "chi2_p-value"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for pvs_id_exp_tuple, pvs_data in pvs_id_exp_grouped_scores:
            pvs_id = pvs_id_exp_tuple[0]
            exp_id = pvs_id_exp_tuple[1]
            row_to_store = {"PVS_id": pvs_id, "Exp": exp_id}

            sample_scores = pvs_data["Score"]
            mos = sample_scores.mean()
            logger.info("MOS of the PVS {} in experiment {}: {:.3f}".format(pvs_id, exp_id, mos))
            row_to_store["MOS"] = mos

            score_counts = get_answer_counts(sample_scores)
            row_to_store["count1"] = score_counts[0]
            row_to_store["count2"] = score_counts[1]
            row_to_store["count3"] = score_counts[2]
            row_to_store["count4"] = score_counts[3]
            row_to_store["count5"] = score_counts[4]

            # est = estimated
            psi_est, rho_est = estimate_parameters(sample_scores, prob_grid_gsd_df)
            logger.info("Estimated psi: {}, estimated rho: {}".format(psi_est, rho_est))
            row_to_store["psi_hat"] = psi_est
            row_to_store["rho_hat"] = rho_est

            chi_squared, p_value = perform_chi_squared(sample_scores, prob_grid_gsd_df, psi_est, rho_est)
            logger.info("p-value of the chi-squared test: {:.4f} (with the chi-squared statistic: {:.4f})"
                        .format(p_value, chi_squared))
            row_to_store["chi2_p-value"] = p_value
            row_to_store["chi2"] = chi_squared

            writer.writerow(row_to_store)

    logger.info("Everything done!")


if __name__ == '__main__':
    main()
    exit(0)
