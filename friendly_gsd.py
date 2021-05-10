# Provides a friendly interface for the GSD model parameters estimation, at the same time giving information about the
# goodness-of-fit of the model to your data. Lastly, it shows how to create p-value P--P plot.
#
# Author: Jakub Nawała <jnawala@agh.edu.pl>

import logging
from _logger import setup_console_and_file_logger
from sys import argv
import argparse
from os import path
from os.path import splitext
import pandas as pd
import numpy as np
import gsd
import bootstrap
from scipy.stats import norm
import matplotlib as mpl
from matplotlib import pyplot as plt
from G_test_on_real_data import read_input_data_subsection
from probability_grid_estimation import get_answer_counts, estimate_parameters, preprocess_real_data
from pathlib import Path

mpl.rcParams["backend"] = "TkAgg"
mpl.rcParams["interactive"] = True
plt.style.use("ggplot")
logger = None


def proces_input_parameters(_argv: list):
    """
    Processes parameters supplied by the user

    :param _argv: a list of argument values to use if this script is not called directly from the CLI
    :return: argparse.Namespace with input parameters processed
    """

    def file_path(string):
        if path.isfile(string):
            return string
        else:
            raise argparse.ArgumentTypeError(f"{string} file does not exist. Please check if you spelled the path "
                                             f"correctly.")

    def positive_int(string):
        if int(string) > 0:
            return int(string)
        else:
            raise argparse.ArgumentTypeError(f"{string} is not a positive integer. Please supply a positive integer.")

    def non_negative_int(string):
        if int(string) >= 0:
            return int(string)
        else:
            raise argparse.ArgumentTypeError(f"{string} is not a non-negative integer. Please supply an integer "
                                             f"greater or equal to 0.")

    parser = argparse.ArgumentParser(description="Provides a friendly interface for the GSD model parameters "
                                                 "estimation. It also gives information about how well"
                                                 " the model fits your data. This is presented in a form of "
                                                 "p-value P–P plot.")
    parser.add_argument("data_csv_filepath", help="a filepath to a tidy CSV file with your data",
                        metavar="data_filepath", type=file_path)
    parser.add_argument("-p", "--pickle", help="a filepath to the pickled probability grid of the GSD model. It "
                                               "defaults to gsd_prob_grid.pkl.", default="gsd_prob_grid.pkl",
                        metavar="path", type=file_path)
    parser.add_argument("-c", "--chunks", help="(for batch processing) the number of chunks into which the input"
                                               " data should be split. It defaults to 1.",
                        metavar="N", default=1, type=positive_int)
    parser.add_argument("-i", "--index", help="(for batch processing) 0-based index of a chunk to process. It default "
                                              "to 0.",
                        metavar="idx", default=0, type=non_negative_int)
    parser.add_argument("-s", "--stimulus-identifier", help="header of a column identifying stimuli. It defaults to"
                                                            " stimulus_id.",
                        metavar="identifier", default="stimulus_id")
    parser.add_argument("-o", "--score-identifier", help="header of a column containing scores. It defaults to"
                                                         " score.",
                        metavar="identifier", default="score")
    parser.add_argument("-e", "--group-also-by-experiment", help="when reading the results group both by stimulus and "
                                                                 "experiment identifiers. This is necessary when "
                                                                 "dealing with data where stimulus identifier is not "
                                                                 "unique across different experiments.",
                        action="store_true")
    parser.add_argument("-f", "--store-figure", help="store the P–P plot on the disk instead of displaying it.",
                        action="store_true")
    if __name__ != '__main__':
        args = parser.parse_args(_argv)
    else:
        args = parser.parse_args()
    return args


def perform_g_test(keys_for_coi: list, data_grouped: pd.core.groupby.GroupBy, prob_grid_gsd: pd.DataFrame,
                   n_bootstrap_samples=10000, score_col_identifier="score", grouped_also_by_experiment=False):
    """
    Perform bootstrapped G-test of goodness-of-fit (GoF) on scores of each stimulus for a given chunk of data (as
    identified by *keys_for_coi*). The G-test assesses how well the Generalized Score Distribution (GSD) fits the
    observable data (available through *data_grouped*). To speed-up the computations the function uses a pre-calculated
    probability grid (*prob_grid_gsd*).

    TODO 2. Allow to choose how the column with p-values is called (or otherwise solve this issue)

    :param keys_for_coi: a list of DataFrameGroupBy keys identifying stimuli relevant for a chunk of interest (coi)
    :param data_grouped: subjective scores grouped by a stimulus identifier
    :param prob_grid_gsd: pre-computed probability grid of observing each answer, assuming the GSD model with a given
     set of parameters
    :param n_bootstrap_samples: the number of bootstrap samples to generate. Higher number means higher p-value
     approximation precision. For example, for the default 10,000, the p-value is approximated with the precision of
     0.0001.
    :param score_col_identifier: a column header identifying the column with scores
    :param grouped_also_by_experiment: a flag indicating whether the keys in *keys_for_coi* are tuples (i.e., input data
     was grouped both by stimulus and experiment identifiers; this flag should be set to True) or not.
    :return: a DataFrame with G-test results (i.e., estimated GSD parameters, T statistic of the test, p-value of the
     test). Importantly, the DataFrame is indexed with stimulus identifiers
    """
    global logger
    if logger is None:  # in case this function is run from outside of this script
        logger = setup_console_and_file_logger(name=__name__, log_file_name=splitext(argv[0])[0] + ".log",
                                               level=logging.INFO)
    logger.info("There are {} stimuli to process".format(len(keys_for_coi)))

    g_test_res = pd.DataFrame(columns=["stimulus_id", "psi_hat", "rho_hat", "T", "p_value", "count1", "count2",
                                       "count3", "count4", "count5"], index=keys_for_coi)

    # Perform the G-test for each stimulus
    it_num = 1  # monitor iteration number
    for key_for_coi in keys_for_coi:
        if grouped_also_by_experiment:
            stimulus_id = key_for_coi[0]
            experiment_id = key_for_coi[1]
            logger.info(f"Processing stimulus number {it_num} from experiment {experiment_id}")
        else:
            stimulus_id = key_for_coi
            logger.info(f"Processing stimulus number {it_num}")
        stimulus_data = data_grouped.get_group(key_for_coi)
        sample_scores = stimulus_data[score_col_identifier]
        score_counts = np.array(get_answer_counts(sample_scores))

        logger.debug("Estimating GSD model parameters")
        psi_hat_gsd, rho_hat = estimate_parameters(sample_scores, prob_grid_gsd)

        logger.debug("Calculating the T test statistic")
        # exp_prob = expected probability
        exp_prob_gsd = gsd.prob(psi_hat_gsd, rho_hat)
        T_statistic_gsd = bootstrap.T_statistic(score_counts, exp_prob_gsd)

        logger.debug("Generating 10k bootstrap samples")
        n_total_scores = np.sum(score_counts)
        bootstrap_samples_gsd = gsd.sample(psi_hat_gsd, rho_hat, n_total_scores, n_bootstrap_samples)

        logger.debug("Estimating GSD model parameters for each bootstrap sample")
        psi_hat_rho_hat_gsd_bootstrap = np.apply_along_axis(estimate_parameters, axis=1, arr=bootstrap_samples_gsd,
                                                            prob_grid_df=prob_grid_gsd, sample_as_counts=True)

        logger.debug("Translating the estimated parameters into probabilities of each answer")

        def _get_each_answer_probability(psi_rho_row, prob_generator):
            """
            Translates psi and rho parameters into the probability of each answer.

            :param psi_rho_row: a 2-column vector with the first col. corresponding to psi and the second one to rho
            :param prob_generator: gsd.prob
            :return: a vector of probabilities of each answer
            """
            psi = psi_rho_row[0]
            rho = psi_rho_row[1]
            return prob_generator(psi, rho)

        bootstrap_exp_prob_gsd = np.apply_along_axis(_get_each_answer_probability, axis=1,
                                                     arr=psi_hat_rho_hat_gsd_bootstrap, prob_generator=gsd.prob)

        logger.debug("Performing the G-test")
        p_value_g_test_gsd = bootstrap.G_test(score_counts, exp_prob_gsd, bootstrap_samples_gsd,
                                              bootstrap_exp_prob_gsd)

        g_test_res.loc[[key_for_coi], "stimulus_id"] = stimulus_id
        g_test_res.loc[[key_for_coi], "psi_hat"] = psi_hat_gsd
        g_test_res.loc[[key_for_coi], "rho_hat"] = rho_hat
        g_test_res.loc[[key_for_coi], "T"] = T_statistic_gsd
        g_test_res.loc[[key_for_coi], "p_value"] = p_value_g_test_gsd
        g_test_res.loc[[key_for_coi], "count1"] = score_counts[0]
        g_test_res.loc[[key_for_coi], "count2"] = score_counts[1]
        g_test_res.loc[[key_for_coi], "count3"] = score_counts[2]
        g_test_res.loc[[key_for_coi], "count4"] = score_counts[3]
        g_test_res.loc[[key_for_coi], "count5"] = score_counts[4]

        it_num += 1

    return g_test_res


def draw_p_value_pp_plot(g_test_res: pd.DataFrame, thresh_pvalue=0.2, should_store_figure=False, ext="pdf",
                         filename_addition="", pval_col_id="p_value"):
    """
    Draws p-value P--P plot for G-test of goodness-of-fit data provided in *g_test_res*. By default, the x-axis of the
    plot spans the range from 0 to 0.2 (cf. *thresh_pvalue*). One can ask to store the resulting plot
    (*should_store_figure*) and specify the related data format (*ext*).

    :param g_test_res: a DataFrame with G-test of goodness-of-fit data. It must contain at least one column (i.e.,
     p_value)
    :param thresh_pvalue: the x-axis spans the range from 0 up to this value
    :param should_store_figure: a flag indicating whether to store plots on the disk
    :param ext: file extension to use when storing figures (e.g., png or pdf)
    :param filename_addition: a string allowing to make the output file's filename unique. (Useful when this function
     is called multiple times for different data.)
    :param pval_col_id: p-value column identifier that should be used when reading from *g_test_res*
    :return: a figure handle
    """
    n_pvs = len(g_test_res)
    p_values = pd.Series(np.linspace(start=0.001, stop=thresh_pvalue, num=100))

    def count_pvs_fraction(p_value, p_value_per_pvs):
        return np.sum(p_value_per_pvs <= p_value) / len(p_value_per_pvs)

    pvs_fraction_gsd = g_test_res[pval_col_id].apply(count_pvs_fraction, args=(g_test_res[pval_col_id],))
    significance_line = p_values + norm.ppf(0.95) * np.sqrt(p_values * (1 - p_values) / n_pvs)

    fig = plt.figure()
    plt.scatter(g_test_res[pval_col_id], pvs_fraction_gsd, label="GSD")
    plt.xlabel("theoretical uniform cdf")
    plt.ylabel("ecdf of $p$-values")
    plt.plot(p_values, significance_line, "-k")
    plt.xlim([0, thresh_pvalue])
    plt.ylim([0, thresh_pvalue + 0.1])
    plt.minorticks_on()

    if should_store_figure:
        saved_fig_filename = "_".join(["p-value_pp-plot", filename_addition]) + "." + ext
        plt.savefig(saved_fig_filename)
        plt.close(fig)
        print(f"Stored the P-P plot in the {saved_fig_filename} file")
    else:
        plt.show()

    return fig


def fit_gsd(scores: pd.Series, gsd_prob_grid_filepath="gsd_prob_grid.pkl"):
    """
    Fits the GSD model to a vector of subjective scores (scores) given the pre-computed probability grid
    (gsd_prob_grid_filepath).

    :param scores: a Pandas Series with subjective scores assigned to a single stimulus
    :param gsd_prob_grid_filepath: a filepath to the pre-computer probability grid for the GSD model
    :return: psi and rho of the fitted GSD model
    """
    assert path.exists(gsd_prob_grid_filepath), f"The file with the probability grid you specified does not exist: " \
                                                f"{gsd_prob_grid_filepath}. Please make sure you typed its path " \
                                                f"correctly."
    gsd_prob_grid = pd.read_pickle(gsd_prob_grid_filepath)
    psi_hat, rho_hat = estimate_parameters(scores, gsd_prob_grid)
    return psi_hat, rho_hat


def main(_argv=None):
    args = proces_input_parameters(_argv)

    # Read the input data in chunks
    n_chunks = args.chunks
    chunk_idx = args.index
    assert n_chunks > 0 and 0 <= chunk_idx < n_chunks
    # Create a logger here to make sure each log has a unique filename (according to a chunk being processed)
    global logger
    logger = setup_console_and_file_logger(name=__name__,
                                           log_file_name=splitext(argv[0])[0] + "_chunk_id_{}_of_{}_chunks.log"
                                           .format(chunk_idx, n_chunks), level=logging.INFO)
    logger.info("Reading chunk with id {} (of {} total chunks)".format(chunk_idx, n_chunks))
    # Read the appropriate chunk of data
    in_csv_filepath = Path(args.data_csv_filepath)
    assert in_csv_filepath.exists() and in_csv_filepath.is_file(), f"Make sure the {in_csv_filepath} file exists"
    data_grouped = preprocess_real_data(in_csv_filepath, should_also_group_by_exp=args.group_also_by_experiment,
                                        stimulus_identifier=args.stimulus_identifier)
    # coi - chunk of interest
    keys_for_coi = read_input_data_subsection(data_grouped, n_chunks, chunk_idx)

    # Perform the bootstrapped G-test of goodness-of-fit on the chunk of data
    # Read the pre-computed probability grid for the GSD model
    prob_grid_gsd = pd.read_pickle(args.pickle)
    # res - results
    g_test_res = perform_g_test(keys_for_coi, data_grouped, prob_grid_gsd,
                                score_col_identifier=args.score_identifier,
                                grouped_also_by_experiment=args.group_also_by_experiment)

    # Visualise G-test results in a form of p-value pp-plot
    in_csv_filename_wo_ext = in_csv_filepath.stem  # wo - without, ex - extension
    if n_chunks > 1:
        logger.info("Since this is the batch processing mode, I am not generating any P–P plots")
        pp_plot_fig_handle = draw_p_value_pp_plot(g_test_res, should_store_figure=args.store_figure,
                                                  filename_addition=in_csv_filename_wo_ext)

    # Store G-test results in a CSV file
    csv_filename = "_".join(["G_test_on", in_csv_filename_wo_ext, f"chunk_id_{chunk_idx}_of_{n_chunks}_chunks.csv"])
    logger.info(f"Storing the results of G-test of goodness-of-fit in the {csv_filename} file")
    g_test_res.to_csv(csv_filename, index=False)
    return


if __name__ == '__main__':
    main()

    logger.info("Everything done!")
    exit(0)
