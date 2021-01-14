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


def reproduce_table_one():
    print("Reproducing Tab. 1")
    prob_grid_gsd = pd.read_pickle("gsd_prob_grid.pkl")
    default_float_format = pd.get_option("float_format")
    pd.set_option("float_format", "{:.3f}".format)
    print(prob_grid_gsd.loc[[(2.1, 0.95), (2.1, 0.88), (2.1, 0.81), (2.1, 0.72), (2.1, 0.61), (2.1, 0.38)]])
    pd.set_option("float_format", default_float_format)
    return


def main():
    # Reproduce the creation of Tab. 1
    reproduce_table_one()
    # TODO 1. Reproduce the creation of Tab. 2
    return


if __name__ == '__main__':
    main()
    exit(0)
