# Performs the G-test to estimate the goodness-of-fit of the GSD and QNormal models to real data.
#
# This script requires two parameters:
#   (i) the number of chunks to cut the input data into and
#   (ii) a zero-based chunk index of a chunk you want to process
#
# Author: Jakub Nawała <jnawala@agh.edu.pl>
# Date: March, 18 2020

import logging
from _logger import setup_console_and_file_logger
from probability_grid_estimation import preprocess_real_data, get_answer_counts, estimate_parameters
import csv
import pandas as pd
import bootstrap
import gsd
import qnormal
import numpy as np
from sys import argv
from pathlib import Path
from data_analysis import estimatate_gsd_parameters

logger = None


def read_input_data_subsection(grouped_scores: pd.core.groupby.GroupBy, n_subsection, subsection_idx):
    """
    Inputs tidy subjective scores grouped by a certain feature (e.g., stimulus ID), splits the scores into
    *n_subsection* subsections and returns DataFrameGroupBy keys for only the i-th subsection, where i is defined by the
    *subsection_idx* parameter.

    :param grouped_scores: scores grouped by a feature defining data granularity
    :param n_subsection: a number saying into how many subsections to divide the input data into
    :param subsection_idx: a zero-based index specifying which subsection of the input to return
    :return: a list of DataFrameGroupBy keys of the selected subsection. Use these to read data relevant for the
     chunk of interest from grouped_scores.
    """
    group_key = list(grouped_scores.groups)

    def chunkify(lst, n):
        """
        See this StackOverflow answer for more details: https://stackoverflow.com/a/2136090/3978083

        :param lst: a list to split into n chunks
        :param n: the number of chunks into which to split the lst list
        :return: a list containing n lists, each being a chunk of the lst list
        """
        return [lst[i::n] for i in range(n)]

    chunked_group_key = chunkify(group_key, n_subsection)
    # coi - chunk of interest
    keys_for_coi = chunked_group_key[subsection_idx]

    return keys_for_coi


def get_each_answer_probability(psi_sigma_row, prob_generator):
    """
    Translates psi and sigma (or rho) parameters into the probability of each answer.

    :param psi_sigma_row: a 2-column vector with the first col. corresponding to psi and the second one to
     sigma (or rho)
    :param prob_generator: either gsd.prob or qnormal.prob
    :return: a vector of probabilities of each answer
    """
    psi = psi_sigma_row[0]
    sigma_or_rho = psi_sigma_row[1]
    return prob_generator(psi, sigma_or_rho)


def main(_argv):
    assert len(_argv) == 4, "This script requires 3 parameters: the number of chunks, a zero-based chunk index and " \
                            "path of a CSV file you wish to process"

    prob_grid_gsd_df = pd.read_pickle("gsd_prob_grid.pkl")
    prob_grid_qnormal_df = pd.read_pickle("qnormal_prob_grid.pkl")

    filepath_cli_idx = 3
    in_csv_filepath = Path(_argv[filepath_cli_idx])
    assert in_csv_filepath.exists() and in_csv_filepath.is_file(), f"Make sure the {_argv[filepath_cli_idx]} file " \
                                                                   f"exists"

    n_chunks_argv_idx = 1
    chunk_idx_argv_idx = 2
    n_chunks = int(_argv[n_chunks_argv_idx])
    chunk_idx = int(_argv[chunk_idx_argv_idx])
    assert n_chunks > 0 and 0 <= chunk_idx < n_chunks

    # Create a logger here to make sure each log has a unique filename (according to a chunk being processed)
    global logger
    logger = setup_console_and_file_logger(name=__name__,
                                           log_file_name="G_test_on_real_data" +
                                                         "_chunk{:03d}_of{:03d}".format(chunk_idx, n_chunks)
                                                         + ".log",
                                           level=logging.DEBUG)

    logger.info("Reading chunk {} (from {} chunks)".format(chunk_idx, n_chunks))
    # coi - chunk of interest
    pvs_id_exp_grouped_scores = preprocess_real_data(in_csv_filepath, should_also_group_by_exp=True)
    keys_for_coi = read_input_data_subsection(pvs_id_exp_grouped_scores, n_chunks, chunk_idx)

    in_csv_filename_wo_ext = in_csv_filepath.stem  # wo - without, ex - extension
    csv_results_filename = "G_test_on_" + in_csv_filename_wo_ext + "_chunk{:03d}_".format(chunk_idx) + \
                           "of_{:03d}".format(n_chunks) + ".csv"
    logger.info("Storing the results in the {} file".format(csv_results_filename))

    with open(csv_results_filename, 'w', newline='', buffering=1) as csvfile:
        fieldnames = ["PVS_id", "count1", "count2", "count3", "count4", "count5", "MOS", "Exp", "psi_hat_gsd",
                      "rho_hat", "psi_hat_qnormal", "sigma_hat", "T_gsd", "T_qnormal", "p-value_gsd", "p-value_qnormal"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # iteration number to make it easier to assess the progress
        it_num = 1
        for pvs_id_exp_tuple in keys_for_coi:
            pvs_id = pvs_id_exp_tuple[0]
            exp_id = pvs_id_exp_tuple[1]
            pvs_data = pvs_id_exp_grouped_scores.get_group(pvs_id_exp_tuple)
            row_to_store = {"PVS_id": pvs_id, "Exp": exp_id}
            logger.info("Iteration {}".format(it_num))
            logger.info("Processing PVS {} from experiment {}".format(pvs_id, exp_id))

            sample_scores = pvs_data["Score"]
            mos = sample_scores.mean()
            logger.info("MOS of the PVS {} in experiment {}: {:.3f}".format(pvs_id, exp_id, mos))
            row_to_store["MOS"] = mos

            score_counts = np.array(get_answer_counts(sample_scores))
            row_to_store["count1"] = score_counts[0]
            row_to_store["count2"] = score_counts[1]
            row_to_store["count3"] = score_counts[2]
            row_to_store["count4"] = score_counts[3]
            row_to_store["count5"] = score_counts[4]

            logger.info("Estimating both models parameters using MLE on the probability grid")
            # est = esimated
            psi_hat_gsd, rho_hat = estimate_parameters(sample_scores, prob_grid_gsd_df)
            psi_hat_qnormal, sigma_hat = estimate_parameters(sample_scores, prob_grid_qnormal_df)
            row_to_store["psi_hat_gsd"] = psi_hat_gsd
            row_to_store["rho_hat"] = rho_hat
            row_to_store["psi_hat_qnormal"] = psi_hat_qnormal
            row_to_store["sigma_hat"] = sigma_hat

            logger.info("Calculating T statistic for both models")
            # exp_prob = expected probability
            exp_prob_gsd = gsd.prob(psi_hat_gsd, rho_hat)
            exp_prob_qnormal = qnormal.prob(psi_hat_qnormal, sigma_hat)
            T_statistic_gsd = bootstrap.T_statistic(score_counts, exp_prob_gsd)
            T_statistic_qnormal = bootstrap.T_statistic(score_counts, exp_prob_qnormal)
            row_to_store["T_gsd"] = T_statistic_gsd
            row_to_store["T_qnormal"] = T_statistic_qnormal

            logger.info("Generating 10k bootstrap samples for both models")
            n_total_scores = np.sum(score_counts)
            n_bootstrap_samples = 10000
            bootstrap_samples_gsd = gsd.sample(psi_hat_gsd, rho_hat, n_total_scores, n_bootstrap_samples)
            bootstrap_samples_qnormal = qnormal.sample(psi_hat_qnormal, sigma_hat, n_total_scores, n_bootstrap_samples)

            # Estimate GSD and QNormal parameters for each bootstrapped sample
            logger.info("Estimating GSD and QNormal parameters for each bootstrapped sample")
            # Use the OpenCL-accelerated GSD estimation
            psi_hat_rho_hat_gsd_bootstrap = estimatate_gsd_parameters(bootstrap_samples_gsd,
                                                                      subsample_size=n_total_scores,
                                                                      gsd_prob_grid_filepath="gsd_prob_grid.pkl")
            psi_hat_sigma_hat_qnormal_bootstrap = np.apply_along_axis(estimate_parameters, axis=1,
                                                                      arr=bootstrap_samples_qnormal,
                                                                      prob_grid_df=prob_grid_qnormal_df,
                                                                      sample_as_counts=True)

            # Translate the estimated bootstrap parameters into probabilities of each answer
            logger.info("Translating the estimated parameters into probabilities of each answer")

            bootstrap_exp_prob_gsd = np.apply_along_axis(get_each_answer_probability, axis=1,
                                                         arr=psi_hat_rho_hat_gsd_bootstrap, prob_generator=gsd.prob)
            bootstrap_exp_prob_qnormal = np.apply_along_axis(get_each_answer_probability, axis=1,
                                                             arr=psi_hat_sigma_hat_qnormal_bootstrap,
                                                             prob_generator=qnormal.prob)

            # Perform the G-test
            logger.info("Performing the G-test")
            p_value_g_test_gsd = bootstrap.G_test(score_counts, exp_prob_gsd, bootstrap_samples_gsd,
                                                  bootstrap_exp_prob_gsd)
            p_value_g_test_qnormal = bootstrap.G_test(score_counts, exp_prob_qnormal, bootstrap_samples_qnormal,
                                                      bootstrap_exp_prob_qnormal)
            row_to_store["p-value_gsd"] = p_value_g_test_gsd
            row_to_store["p-value_qnormal"] = p_value_g_test_qnormal
            logger.info("p-value (G-test) for GSD: {}".format(p_value_g_test_gsd))
            logger.info("p-value (G-test) for QNormal: {}".format(p_value_g_test_qnormal))

            writer.writerow(row_to_store)
            it_num += 1


if __name__ == '__main__':
    main(argv)

    logger.info("Everything done!")

    exit(0)
