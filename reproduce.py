# A master script allowing to reproduce the results from the following ACM MM'20 paper:
#
# @inproceedings{Nawała2020,
#     author = {Nawa{\l}a, Jakub and Janowski, Lucjan and {\'{C}}miel, Bogdan and Rusek, Krzysztof},
#     booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (MM ’20)},
#     keywords = {experiment consistency,experiment validity,quality of experience,subjective experiment},
#     title = {{Describing Subjective Experiment Consistency by p-Value P–P Plot}},
#     year = {2020}
# }
#
# Author: Jakub Nawała <jnawala@agh.edu.pl>
# Created: Jan 14, 2021

import pandas as pd
from enum import Enum
import numpy as np
from scipy.stats import norm
from friendly_gsd import draw_p_value_pp_plot
import friendly_gsd
import argparse
import G_test_on_real_data


class Scenario(Enum):
    """
    Makes it easier to refer to a specific execution scenario
    """
    USE_EXISTING_RES = 1
    FOR_FIG_THREE_ONLY = 2
    REPRODUCE_ALL = 3


class Experiment(Enum):
    """
    An enumeration unifying the access interface to experiment name and ID
    """
    HDTV1 = 1
    HDTV2 = 2
    HDTV3 = 3
    HDTV4 = 4
    HDTV5 = 5
    HDTV6 = 6
    ITS4S_NTIA = 7
    ITS4S_AGH = 8
    AGH_NTIA = 9
    MM2_NTIA_LAB = 10
    MM2_NTIA_CAFETERIA = 11
    MM2_INTEL_LAB = 12
    MM2_IRCCYN_LAB = 13
    MM2_IRCCYN_CAFETERA = 14
    MM2_TECHNICOLOR_LAB = 15
    MM2_TECHNICOLOR_PATIO = 16
    MM2_AGH_LAB = 17
    MM2_AGH_HALLWAY = 18
    MM2_OPTICOM_HOME = 19
    ITS4S2 = 20
    ITU_T_SUPP23 = 21


def check_consistency_of_all_experiments(g_test_res_csv_filepath="G_test_results.csv", conj_alpha=0.2,
                                         should_store_as_csv=False):
    """
    Uses hypothesis testing to classify a given experiment as consistent or inconsistent. This corresponds to the
    analysis described in the original paper, but with alpha being constant and equal to *conj_alpha*.

    Null hypothesis: The fraction of stimuli with score distribution not compliant with the GSD is not greater than
    *conj_alpha*. (Note that we treat as not compliant with the GSD all stimuli, which score distribution get
    assigned by the G-test of goodness-of-fite a p-value below *conj_alpha*.)

    Alternative hypothesis: The fraction of stimuli with score distribution not compliant with the GSD is greater than
    *conj_alpha*.

    WARNING: Treat the results of this analysis as a preliminary view on experiment consistency. Generate a P–P plot
    to get the complete view of the situation.

    :param g_test_res_csv_filepath: a filepath of the CSV file with G-test results
    :param conj_alpha: conjectured alpha (i.e., the fraction of stimuli that do not follow the GSD)
    :param should_store_as_csv: a flag indicating whether to store the results in a CSV file (named
     consistency_hypothesis_check_res.csv)
    :return: a DataFrame with p-values for each experiment
    """
    g_test_results = pd.read_csv(g_test_res_csv_filepath)

    # consist_hyp_res - consistency hypothesis check results
    consist_hyp_res = pd.DataFrame(np.zeros((len(Experiment), 2)), columns=["pvalue", "is_consistent"],
                                   index=list(Experiment))

    print(f"Conjectured alpha (i.e., the fraction of stimuli that do not follow the GSD): {conj_alpha}")

    for exp_enum in Experiment:
        print("-" * 10)
        print("Processing experiment {}".format(exp_enum.name))
        exp_id = exp_enum.value
        exp_res = g_test_results.groupby("Exp").get_group(exp_id)

        # theor_thresh - theoretical threshold, n - the number of
        theor_thresh = conj_alpha + 1.64 * np.sqrt(conj_alpha * (1 - conj_alpha) / len(exp_res))
        n_pvs_below_thresh = np.sum(exp_res["p-value_gsd"] < conj_alpha)
        n_pvs = len(exp_res)
        observed_alpha = n_pvs_below_thresh / n_pvs

        if observed_alpha > theor_thresh:
            print("The experiment is inconsistent")
            consist_hyp_res.loc[exp_enum, "is_consistent"] = False
        else:
            print("The experiment is consistent")
            consist_hyp_res.loc[exp_enum, "is_consistent"] = True

        # Find the p-value of the test
        # Calculate the test statistic (which comes from the standard normal distribution under H0)
        test_statistic = (n_pvs_below_thresh - n_pvs * conj_alpha) / (np.sqrt(n_pvs * conj_alpha * (1 - conj_alpha)))
        # Compare it to the standard normal distribution (specifically, its CDF)
        pvalue = 1 - norm.cdf(test_statistic)
        print("p-value: {:.4f}".format(pvalue))

        consist_hyp_res.loc[exp_enum, "pvalue"] = pvalue

    if should_store_as_csv:
        consist_hyp_res.to_csv("consistency_hypothesis_check_res.csv")
    return consist_hyp_res


def reproduce_table_one():
    print("Reproducing Tab. 1")
    print("=" * 18)
    prob_grid_gsd = pd.read_pickle("gsd_prob_grid.pkl")
    default_float_format = pd.get_option("float_format")
    pd.set_option("float_format", "{:.3f}".format)
    # df - DataFrame; oi - of interest
    df_oi = prob_grid_gsd.loc[[(2.1, 0.95), (2.1, 0.88), (2.1, 0.81), (2.1, 0.72), (2.1, 0.61), (2.1, 0.38)]]
    print(df_oi)
    pd.set_option("float_format", default_float_format)
    df_oi_float = df_oi.astype(float)
    out_csv_filename = "table_one_score_distribution.csv"
    df_oi_float.to_csv(out_csv_filename, float_format="%.3f")
    print(f"Stored Tab. 1 in the {out_csv_filename} file")
    return


def reproduce_table_two(g_test_res_csv_filepath="G_test_results.csv"):
    """
    Reproduces Tab. 2 from the original paper.

    :param g_test_res_csv_filepath: a filepath to a file with G-test results to use
    :return: nothing
    """
    print("\nRunning computations necessary for reproduction of Tab.2...\n")
    # pval - p-value; exp - experiment
    pval_per_exp = check_consistency_of_all_experiments(g_test_res_csv_filepath)
    print("\nReproducing Tab. 2")
    print("=" * 18)
    default_float_format = pd.get_option("float_format")
    pd.set_option("float_format", "{:.5f}".format)
    # ser - Pandas Series; oi - of interest
    ser_oi = pval_per_exp.loc[[Experiment.ITS4S2, Experiment.ITS4S_AGH, Experiment.ITS4S_NTIA,
                               Experiment.MM2_IRCCYN_LAB], "pvalue"]
    print(ser_oi)
    pd.set_option("float_format", default_float_format)
    out_csv_filename = "table_two_pvals.csv"
    ser_oi.to_csv(out_csv_filename, float_format="%.5f", index_label="Experiment")
    print(f"Stored Tab. 2 in the {out_csv_filename} file")
    return


def reproduce_figure_three(g_test_res_csv_filepath="G_test_results.csv"):
    """
    Reproduces Fig. 3 from the original paper.

    :param g_test_res_csv_filepath: a filepath to a file with G-test results to use
    :return: nothing
    """
    print("\nReproducing Fig. 3")
    print("=" * 18)
    # Read the G-test of GoF p-values relevant for the analysis. res - results
    g_test_results = pd.read_csv(g_test_res_csv_filepath)
    hdtv1_res = g_test_results.groupby("Exp").get_group(Experiment.HDTV1.value)
    its4s2_res = g_test_results.groupby("Exp").get_group(Experiment.ITS4S2.value)
    its4s_res = g_test_results.groupby("Exp").get_group(Experiment.ITS4S_AGH.value)
    agh_ntia_res = g_test_results.groupby("Exp").get_group(Experiment.AGH_NTIA.value)
    # Generate P–P plots and store them on the disk
    draw_p_value_pp_plot(hdtv1_res, should_store_figure=True,
                         filename_addition="_".join([Experiment.HDTV1.name, "fig_three"]), pval_col_id="p-value_gsd")
    draw_p_value_pp_plot(its4s2_res, should_store_figure=True,
                         filename_addition="_".join([Experiment.ITS4S2.name, "fig_three"]), pval_col_id="p-value_gsd")
    draw_p_value_pp_plot(its4s_res, should_store_figure=True,
                         filename_addition="_".join([Experiment.ITS4S_AGH.name, "fig_three"]),
                         pval_col_id="p-value_gsd")
    draw_p_value_pp_plot(agh_ntia_res, should_store_figure=True,
                         filename_addition="_".join([Experiment.AGH_NTIA.name, "fig_three"]), pval_col_id="p-value_gsd")
    return


def reproduce_table_three(g_test_res_csv_filepath="G_test_results.csv"):
    """
    Reproduces Tab. 3 from the original paper.

    :param g_test_res_csv_filepath: a filepath to a file with G-test results to use
    :return: nothing
    """
    print("\nReproducing Tab. 3")
    print("=" * 18)
    # Read five stimuli with the lowest p-value from the ITS4S_AGH experiment
    g_test_results = pd.read_csv(g_test_res_csv_filepath)
    its4s_res = g_test_results.groupby("Exp").get_group(Experiment.ITS4S_AGH.value)
    # Sort the results according to GSD p-value
    sorted_its4s_res = its4s_res.sort_values(by="p-value_gsd")
    # Take results for the 5 lowest p-values and store in a CSV file
    five_lowest_its4s = sorted_its4s_res.head(n=5)
    # Change the index from numerical to a, b, c, d, e. li - letter index
    li_five_lowest_its4s = five_lowest_its4s.set_index(pd.Index(['a', 'b', 'c', 'd', 'e']))
    # coi - columns of interest
    li_five_lowest_its4s_coi = li_five_lowest_its4s[["count1", "count2", "count3", "count4", "count5", "p-value_gsd"]]
    print(li_five_lowest_its4s_coi)
    out_csv_filename = "table_three_five_lowest_pvalue_res_its4s_agh.csv"
    li_five_lowest_its4s_coi.to_csv(out_csv_filename, index_label="ID")
    print(f"Stored the table in the {out_csv_filename} file")
    return


def reproduce_table_four(g_test_res_csv_filepath="G_test_results.csv"):
    """
    Reproduces Tab. 4 from the original paper.

    :param g_test_res_csv_filepath: a filepath to a file with G-test results to use
    :return: nothing
    """
    print("\nReproducing Tab. 4")
    print("=" * 18)
    # Read five stimuli with the lowest p-value from the ITS4S2 experiment
    g_test_results = pd.read_csv(g_test_res_csv_filepath)
    its4s2_res = g_test_results.groupby("Exp").get_group(Experiment.ITS4S2.value)
    # Sort the results according to GSD p-value
    sorted_its4s2_res = its4s2_res.sort_values(by="p-value_gsd")
    # Take results for the 5 lowest p-values and store in a CSV file
    five_lowest_its4s2 = sorted_its4s2_res.head(n=5)
    # Change the index from numerical to f, g, h, i, j. li - letter index
    li_five_lowest_its4s2 = five_lowest_its4s2.set_index(pd.Index(['f', 'g', 'h', 'i', 'j']))
    # coi - columns of interest
    li_five_lowest_its4s2_coi = li_five_lowest_its4s2[["count1", "count2", "count3", "count4", "count5", "p-value_gsd"]]
    print(li_five_lowest_its4s2_coi)
    out_csv_filename = "table_four_five_lowest_pvalue_res_its4s2.csv"
    li_five_lowest_its4s2_coi.to_csv(out_csv_filename, index_label="ID")
    print(f"Stored the table in the {out_csv_filename} file")
    return


def reproduce_g_test_results_and_fig_three():
    print("\nReproducing G-test p-values")
    print("=" * 27)
    print("(and generating related P–P plots—--cf. Fig. 3 in the original paper)")
    print("Processing the AGH/NTIA experiment...")
    friendly_gsd.main(["-s", "PVS_id", "-o", "Score", "-e", "-f", "reproducibility/agh_ntia_results.csv"])
    print("Processing the HDTV1 experiment...")
    friendly_gsd.main(["-s", "PVS_id", "-o", "Score", "-e", "-f", "reproducibility/hdtv1_results.csv"])
    print("Processing the ITS4S2 experiment...")
    friendly_gsd.main(["-s", "PVS_id", "-o", "Score", "-e", "-f", "reproducibility/its4s2_results.csv"])
    print("Processing the ITS4S experiment...")
    friendly_gsd.main(["-s", "PVS_id", "-o", "Score", "-e", "-f", "reproducibility/its4s_agh_results.csv"])
    return


def process_input_parameters():
    """
    Processes parameters supplied by the user

    :return: argparse.Namespace with input parameters processed
    """

    def positive_int(string):
        if int(string) > 0:
            return int(string)
        else:
            raise argparse.ArgumentTypeError(f"{string} is not a positive integer. Please supply a positive integer.")

    parser = argparse.ArgumentParser(description="Allows to reproduce all the experiments in the Nawała et al. "
                                                 "Describing Subjective Experiment Consistency by p-Value P-P Plot "
                                                 "paper from ACM MM'20.")
    parser.add_argument("scenario", type=positive_int, help="a digit (1–3) corresponding to an execution scenario of "
                                                            "choice: (1) Redraw and reproduce figures and tables using"
                                                            " the existing G-test results. (2) Reproduce only these"
                                                            " G-test results that are necessary for Fig. 3."
                                                            " (3) Reproduce all G-test results.\n\n"
                                                            "Importantly, scenario 1 needs almost no time to run. "
                                                            "Scenario 2 needs around 224 hours (more than 9 days), "
                                                            "whereas scenario 3 needs around 509 hours (more than "
                                                            "21 days). Read about the batch processing capability "
                                                            "to deal with these long execution times.")
    args = parser.parse_args()
    assert args.scenario < 4, "Please choose either scenario 1, 2 or 3."
    return args


def main():
    args = process_input_parameters()
    if args.scenario == Scenario.REPRODUCE_ALL.value:
        print("Reproducing G-test results for all the 21 experiments...")
        # The first argument is empty since it is not used, but must be there. The second argument asks to split the
        # input data into one chunk only (i.e., it asks not to split the data). The third argument asks to process the
        # chunk at 0-th index. The last argument specifies a CSV file with subjective responses from which to read the
        # input data.
        G_test_on_real_data.main(["", "1", "0", "subjective_quality_datasets.csv"])
        g_test_res_csv_filepath = "G_test_on_subjective_quality_datasets_chunk000_of_001.csv"
    else:
        g_test_res_csv_filepath = "G_test_results.csv"
    reproduce_table_one()  # Does not depend on the choice of the scenario
    reproduce_table_two(g_test_res_csv_filepath)
    if args.scenario == Scenario.USE_EXISTING_RES.value or args.scenario == Scenario.REPRODUCE_ALL.value:
        reproduce_figure_three(g_test_res_csv_filepath)
    else:  # Scenario.FOR_FIG_THREE_ONLY
        reproduce_g_test_results_and_fig_three()
    # TODO 1. When reproducing Fig. 3 G-test results use them with Tab. 3 and 4 as well
    reproduce_table_three(g_test_res_csv_filepath)
    reproduce_table_four(g_test_res_csv_filepath)
    return


if __name__ == '__main__':
    main()
    exit(0)
