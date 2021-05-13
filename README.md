# About

This repository stores data, source codes and an in-depth tutorial all related with
generating *p*-value P&ndash;P plots to assess consistency of subjective data. It
also provides means to reproduce the results presented in [Nawała2020]. (For more
information regarding reproducibility please go to the
[Reproducibility](#reproducibility) section.)

The content of this repository is complementary to the following article. Please cite it if you
make any use of this repository's content.
```
@inproceedings{Nawała2020,
    author = {Nawa{\l}a, Jakub and Janowski, Lucjan and {\'{C}}miel, Bogdan and Rusek, Krzysztof},
    booktitle = {Proceedings of the 28th ACM International Conference on Multimedia (MM ’20)},
    keywords = {experiment consistency,experiment validity,quality of experience,subjective experiment},
    title = {{Describing Subjective Experiment Consistency by p-Value P–P Plot}},
    year = {2020}
}
``` 

Long story short, [the `friendly_gsd.py` script](friendly_gsd.py) is the most important piece
of code here. To learn how to use it (especially if you want to generate *p*-value P&ndash;P plots
for your subjective data) please refer to the
[p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file (cf. the
[In-depth Tutorial about Generating *p*-Value P–P Plots for Your Subjective
Data](#in-depth-tutorial-about-generating-p-value-pp-plots-for-your-subjective-data) section).

To learn how to run our code go to section [Running the Software](#running-the-software).

To get familiar with repository's structure please refer to section [Repository Structure](#repository-structure).

If you want to execute the G-test without generating *p*-value P&ndash;P plot please refer
to section [Run Only the G-test](#run-only-the-g-test).

If you just want to fit the Generalized Score Distribution (GSD) [Janowski2019] model to your
subjective data please go to section [Only Fit the GSD](#only-fit-the-gsd).

If your goal is to learn more about the method, the code or you would like to implement it
from the scratch then please refer to the [In-depth Tutorial about Generating *p*-Value P–P Plots for Your Subjective
Data](#in-depth-tutorial-about-generating-p-value-pp-plots-for-your-subjective-data) section.

To learn more about code's batch processing functionality please see the [Batch Processing](#batch-processing) section.

# Repository Structure

The most important file in this repository is [the `friendly_gsd.py` script](friendly_gsd.py).
It implements *p*-value P&ndash;P plot generation along with all the related functionalities
(i.e., GSD distribution parameters estimation and execution of the bootstrapped version of the
G-test of goodness-of-fit). For more information on how to use this script please refer
to the
[p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file.

[The subjective_quality_datasets.csv file](subjective_quality_datasets.csv) contains subjective
scores from 21 real-life subjective experiments. For more details please refer to [Nawała2020].

[The G_test_results.csv file](G_test_results.csv) contains results of the G-test of goodness-of-fit
(for the fitted GSD distribution) for all stimuli listed in
[the subjective_quality_datasets.csv file](subjective_quality_datasets.csv). To learn what is
the meaning of this file please refer to [Nawała2020] and the
[p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file.

[The hdtv1_exp1_scores_pp_plot_ready.csv file](hdtv1_exp1_scores_pp_plot_ready.csv) contains
subjective scores from the first experiment of the HDTV Phase I subjective study [Pinson2010].
These scores are used in the
[p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file to present what input data format is accepted by the `friendly_gsd.py` script.

[The `friendly_gsd.py` script](friendly_gsd.py) depends on several auxiliary scripts. These are:
`gsd.py`, `bootstrap.py`, `G_test_on_real_data.py`, `probability_grid_estimation.py`,
`_logger.py` and `qnormal.py`.

[The *figures* folder](figures) contains graphics used in the
[p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file. They help explain how to create and interpret P&ndash;P plots.

[The requirements.txt file](requirements.txt) lists PIP packages required to use the code.

[The gsd_prob_grid.pkl file](gsd_prob_grid.pkl) contains pre-calculated probabilities of each
score for a range of psi and rho parameter values. (Psi and rho are the two parameters defining
the shape of the GSD distribution.)

# Running the Software
We show here how to use our software to generate a *p*-value P–P plot (and related G-test results) for exemplary subjective data. We choose to use here the subjective data from the first experiment of the VQEG HDTV Phase I subjective study [Pinson2010].

Assuming the code is run from the root folder of this repository, the following line executed in the terminal
results in a *p*-value P–P plot for the data contained in [the `hdtv1_exp1_scores_pp_plot_ready.csv`
file](hdtv1_exp1_scores_pp_plot_ready.csv):
```
python3 friendly_gsd.py hdtv1_exp1_scores_pp_plot_ready.csv
```

Executing the script this way we ask it to process all 168 stimuli from the first experiment of the HDTV Phase
I test. **Please be advised that this may take long time to finish**. Our internal trials on a consumer-grade laptop (as
of 2020) indicate that it takes around 5 minutes to process a single stimulus
(which corresponds to 14 hours of processing for the entire experiment).
This is why the script supports batch processing (see below for more details). 

## Batch Processing
**<font color=red>TODO</font>**: Copy here the details regarding how to use the batch processing
functionality.

Since performing the bootstrapped version of the G-test of goodness-of-fit (GoF) is computationally-intensive we
designed our code to be batch processing ready. Differently put, you can run multiple instances of the
`friendly_gsd.py` script, each processing a separate part of your input data.

Below is an excerpt from `friendly_gsd.py`'s command-line help message.
```shell
usage: friendly_gsd.py [-h] [-p path] [-c N] [-i idx] [-s identifier]
                       [-o identifier] [-e] [-f]
                       data_filepath
```
In terms of batch processing the `-c` and `-i` optional arguments are of our interest. Here are their help messages.
```
  -c N, --chunks N      (for batch processing) the number of chunks into which
                        the input data should be split. It defaults to 1.
  -i idx, --index idx   (for batch processing) 0-based index of a chunk to
                        process. It default to 0.
```
Using these you can start many instances of `friendly_gsd.py`, each with **the same** input data file, **the same** number
of chunks (`-c`) and each with a **unique** index of a chunk to process (`-i`).

For example, if you have three machines (A, B and C), each with one CPU, it makes sense to start three instances of the
`friendly_gsd.py` script&mdash;one on each of the machines. Now, assuming you have an exact copy of your input
data file (say, `my_input_data.csv`) on each of the machines, here is how you should start the `friendly_gsd.py`
script on each machine.

Machine A
```shell
python3 friendly_gsd.py -c 3 -i 0 my_input_data.csv
```

Machine B
```shell
python3 friendly_gsd.py -c 3 -i 1 my_input_data.csv
```

Machine C
```shell
python3 friendly_gsd.py -c 3 -i 2 my_input_data.csv
```

Please note how the value of the `-i` parameter is changing depending on the machine the script is run on.

After all the computations are finished you end up with one CSV file with G-test results on each machine.
Specifically, on machine A you will get `G-test_chunk_id_0_of_3_chunks.csv`; on machine B:
`G-test_chunk_id_1_of_3_chunks.csv` and on machine C: `G-test_chunk_id_2_of_3_chunks.csv`. These three files
(when combined) contain G-test results for all stimuli present in the `my_input_data.csv` file.

### _p_-Value P–P Plot 

Importantly, running the code in the batch processing mode suppresses the generation
of _p_-value P–P plots. You have to manually generate the P–P plot from the results
by following the steps below.
1. Merge the G-test results generated by all the instances of the `friendly_gsd.py`
script.
   1. Put all the result chunks (e.g., `G-test_chunk_id_0_of_3_chunks.csv`,
      `G-test_chunk_id_1_of_3_chunks.csv` and `G-test_chunk_id_2_of_3_chunks.csv`)
      in one folder (e.g., `chunks`). Make sure to place this folder in the root
      folder of this repository.
   2. Run the following in a Python console (started inside the root folder of
      this repository) to create a `combined.csv` file
   with results from all the chunks combined.
      ```python
      import pandas as pd
      from os import path, listdir
      import re
      from friendly_gsd import read_results_from_chunks
      
      combined_df = read_results_from_chunks("chunks")  # df - DataFrame
      combined_df.to_csv("combined.csv", index=False)
      ```
2. Use the CSV file with all the results combined to generate a _p_-value P–P plot.
   1. Assuming that the CSV file with all the G-test results combined is called
      `combined.csv`, run the following in a Python console
      (started inside the root folder of this repository) to create a _p_-value
      P–P plot and store it on the disk in the `p-value_pp-plot.pdf` file.
      ```python
      import pandas as pd
      from friendly_gsd import draw_p_value_pp_plot
      
      g_test_res = pd.read_csv("combined.csv")
      draw_p_value_pp_plot(g_test_res, should_store_figure=True)
      ```
      
In case of any doubts please consult the documentation of the
`friendly_gsd.read_results_from_chunks` and `friendly_gsd.draw_p_value_pp_plot`
functions.

# Run Only the G-test
In case you are here just to run the G-test of goodness-of-fit (GoF) this section is for you.
To limit `friendly_gsd.py` script's functionality to G-test execution only please comment-out
the call to the `draw_p_value_pp_plot()` function. This is the related code fragment from
`friendly_gsd.py`'s `main()` function.
```python
# Visualise G-test results in a form of p-value pp-plot
pp_plot_fig_handle = draw_p_value_pp_plot(g_test_res)
``` 

# Only Fit the GSD
If your only goal is to fit the Generalized Score Distribution (GSD) [Janowski2019] to a sample
of subjective scores then this section is for you. The `friendly_gsd.py` script contains a
dedicated function for this purpose. It is called `fit_gsd`. It takes two arguments:
1. a vector of subjective scores assigned to a single stimulus of interest (this is expressed as Pandas Series) and
2. a filepath of the Pickle file with the pre-calculated probability grid for the GSD model (this is expressed as a simple string)<sup>1</sup>.

The function returns two numerical values: (i) psi and (ii) rho of the fitted GSD model.

Please note that the `fit_gsd` function is not part of the main script's workflow. This means that to use this
function you will have to write your own script and import the function as follows.
```python
from friendly_gsd import fit_gsd
```
This also means that you will not be able to take advantage of script's batch processing functionality (see section
[Batch Processing](#batch-processing)). The lack of batch processing should not be a problem here since fitting the GSD
works very fast. However, in case you really want to use batch processing please read the 
[Workaround to Retain Batch Processing](#workaround-to-retain-batch-processing) section.

<sup>1</sup> Please note that this file is provided in the repository (cf. [gsd_prob_grid.pkl](gsd_prob_grid.pkl)).

## Workaround to Retain Batch Processing
The relevant code of the `friendly_gsd.py`
script is present in the `friendly_gsd.perform_g_test()` function. There, the following line
fits the GSD model (through the MLE approach) to a sample of subjective scores (expressed
as Pandas Series object `sample_scores`).
```python
psi_hat_gsd, rho_hat = estimate_parameters(sample_scores, prob_grid_gsd)
```
When this line is executed variables `psi_hat_gsd` and `rho_hat` store estimated parameter
values of psi and rho parameters of the GSD distribution, respectively.

The `prob_grid_gsd` represents the pre-calculated grid of probabilities of observing each score
(when the GSD model is assumed) for a range of psi and rho parameter values.

Although the `friendly_gsd.py` script does not directly provide an interface to limit
its operation to GSD model fitting, there is a workaround to achieve this. You have to follow
this two-step procedure:
1. Comment-out the line from `friendly_gsd.py`'s `main()` function responsible for drawing *p*-value P&ndash;P plot:`pp_plot_fig_handle = draw_p_value_pp_plot(g_test_res)`. 
2. Add the `n_bootstrap_samples=1` argument to the call to the `perform_g_test()` function inside `friendly_gsd.py`'s `main()` function.

After the modifications the relevant code fragment should look as follows.
```python
# res - results
g_test_res = perform_g_test(keys_for_coi, data_grouped, prob_grid_gsd,
                            score_col_identifier=args.score_identifier,
                            grouped_also_by_experiment=args.group_also_by_experiment,
                            n_bootstrap_samples=1)

# Visualise G-test results in a form of p-value pp-plot
# pp_plot_fig_handle = draw_p_value_pp_plot(g_test_res)
```

When you now run the `friendly_gsd.py` script it will still perform the G-test, but the overhead
of doing so will be small. (This is because we ask the code to use one and not 10,000 bootstrap
samples when performing the bootstrapped G-test of goodness-of-fit.) When you execute the code
the G-test results and, more importantly, MLE-fitted GSD parameters will be available in
the output CSV file. 

If you do not use the batch processing functionality (see [the
"Batch Processing" section](#batch-processing)) and run the `friendly_gsd.py` script as follows:
```shell script
python3 friendly_gsd.py your_subjective_data.csv
``` 
the output CSV file will be `G-test_chunk_id_0_of_1_chunks.csv`.

For the details on how this output CSV file is formatted and how to prepare your subjective
data for processing with `friendly_gsd.py` please refer to sections "Output" and "Input"
of the [p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file, respectively.

**WARNING**: Although the code run this way returns G-test results these are not meaningful and
should be discarded. For the explanation why this is the case please refer to the
[p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file.

# In-depth Tutorial about Generating *p*-Value P–P Plots for Your Subjective Data

If you have your subjective data and would like to generate a *p*-value P–P plot based on its contents then you are in
the right place. This section will walk you through the process. It contains both a high-level description of the
method, as well as code fragments from the [friendly_gsd.py](friendly_gsd.py) script. These two things combined should
give you a quite solid understanding of the methodology. The section also describes issues related with using
the code in this repository&mdash;how to run it, what input does it require, what output it produces and how to use its
batch processing functionality. 

In order not to make this README too lengthy we provide the tutorial in the form of a Jupyter Notebook.
Please open the [p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb)
file to get access to the tutorial.

If you prefer to view the tutorial locally, then please install the `notebook` PIP package.
With the package installed, run the following (from the repo's root folder) to open the tutorial locally.
```shell script
jupyter notebook p_value_pp_plot_in-depth_tutorial.ipynb
```

**TIP**: In order to more easily navigate through the provided Jupyter Notebook please consider installing and
configuring [the jupyter_contrib_nbextensions PIP package](https://pypi.org/project/jupyter-contrib-nbextensions/).
Among other things, it provides a handy functionality of floating table of contents.

## Limitations

The method works only for subjective data expressed on the 5-level Absolute Category Rating (ACR) scale (cf. Sec. 6.1
of [Rec. ITU-T P.910](https://www.itu.int/rec/T-REC-P.910/en)).

# Batch Processing
Since performing the bootstrapped version of the G-test of goodness-of-fit (GoF) is computationally-intensive we
designed our code to be batch processing ready. Differently put, you can run multiple instances of the
`friendly_gsd.py` script, each processing a separate part of your input data.

For more details on this subject please take a look at the "Batch Processing" section
of the [p_value_pp_plot_in-depth_tutorial.ipynb](https://nbviewer.jupyter.org/github/Qub3k/subjective-exp-consistency-check/blob/master/p_value_pp_plot_in-depth_tutorial.ipynb) file.

# Reproducibility
The secondary purpose of this repository is to provide means to reproduce the
results presented in [Nawała2020]. To this end we make available the `reproduce.py`
script. Its functionality is best explained by its help message.
```
usage: reproduce.py [-h] [-n N] scenario

Allows to reproduce all the experiments in the Nawała et al. Describing
Subjective Experiment Consistency by p-Value P-P Plot paper from ACM MM'20.

positional arguments:
  scenario              a digit (1–5) corresponding to an execution scenario
                        of choice: (1) Redraw and reproduce figures and tables
                        using the existing G-test results. (2) Reproduce only
                        these G-test results that are necessary for Fig. 3.
                        (3) Reproduce all G-test results. Importantly,
                        scenario 1 needs almost no time to run. Scenario 2
                        needs around 224 hours (more than 9 days), whereas
                        scenario 3 needs around 509 hours (more than 21 days).
                        (Read about the batch processing capability to deal
                        with these long execution times.) (4) Run the G-test
                        for N randomly selected stimuli. (5) Reproduce
                        probability grids for the GSD and QNormal models.
                        (This takes about an hour.)

optional arguments:
  -h, --help            show this help message and exit
  -n N, --number-of-stimuli N
                        run the G-test for this many stimuli (only relevant
                        when used in conjunction with scenario 4)
```

## Requirements
Before you run the `reproduce.py` script please ensure that all the packages listed
in the [requirements.txt](requirements.txt) file are installed. You can do so by
running the following in the terminal. (**We recommend to run this command in
a newly created Python virtual environment.**)
```bash
$ python3 -m pip install -r requirements.txt
```

The code was tested with Python 3.7. Thus, our recommendation is to use Python 3.7
or newer.

## Test the Setup
To quickly check that everything is set up properly please run the following in
the terminal.
```bash
$ python3 reproduce.py 1
```

This will reproduce all the figures and tables from [Nawała2020] using data
available in this repository. In other words, this recreates the tables and
figures, but instead of doing that using raw subjective responses, it uses
results derived from those. More specifically, it uses G-test results from
the [G_test_results.csv](G_test_results.csv) file and CSV files available
in the [reproducibility](reproducibility) folder.

## Reproduce Everything From the Scratch
To entirely reproduce the results from [Nawała2020] please run the following
in the terminal.
```bash
$ python3 reproduce.py 3  # Reproduce G-test results and regenerate tables and figures
$ python3 reproduce.py 5  # Reproduce probability grids
```

Please be advised that the first of the two calls to `reproduce.py` **takes about
21 days to finish**. This was tested on the following hardware setup:
Intel Core i3-8130U CPU, 16 GB of 2400 MHz RAM and 256 GB SSD disk
(Lenovo LENSE30256GMSP34MEAT3TA).

If you do not want to wait 21 days to generate G-test results please consult
the [Batch Processing](#batch-processing) section. Having at hand the
G-test results from multiple batch processing chunks please merge them into
on large CSV file. Then, rename the file to `G_test_results.csv`, place it
in the root folder of this repository and run the following to reproduce all
the figures and tables from [Nawała2020] using the reproduced G-test results.
```bash
$ python3 reproduce.py 1
```

Since running the G-test implicitly uses pre-calculated probability grids you
need to reproduce these as well to claim that the complete reproducibility has
been achieved. This is done by the `$ python3 reproduce.py 5` call. Once
it finishes running (which takes about one hour) you end up with two files:
(i) `reproduced_gsd_prob_grid.pkl`
and (ii) `reproduced_qnormal_prob_grid.pkl`. To make sure the two reproduced
probability grids are used whenever you run the G-test, please rename the two
files to `gsd_prob_grid.pkl` and `qnormal_prob_grid.pkl`, respectively. At last,
place the two in the root folder of this repository. From now on, whenever you
run `friendly_gsd.py` or `reproduce.py` scripts (in the mode running the G-test)
they will use the reproduced probability grids.

## Interpreting the Outputs
The listing below shows files generated after running `reproduce.py` with execution
scenarios 3 and 5.
```text
p-value_pp-plot_fig_one_a.pdf
p-value_pp-plot_fig_one_b.pdf
table_one_score_distribution.csv
table_two_pvals.csv
p-value_pp-plot_HDTV1_fig_three.pdf
p-value_pp-plot_ITS4S2_fig_three.pdf
p-value_pp-plot_ITS4S_AGH_fig_three.pdf
p-value_pp-plot_AGH_NTIA_fig_three.pdf
table_three_five_lowest_pvalue_res_its4s_agh.csv
table_four_five_lowest_pvalue_res_its4s2.csv
G_test_on_subjective_quality_datasets_chunk000_of_001.csv
reproduced_gsd_prob_grid.pkl
reproduced_qnormal_prob_grid.pkl
```

The way in which you can check whether the reproduced results are
in line with the original results depends on the output type. For figures
(i.e., all the PDF files),
you have to perform a visual comparison with those in [Nawała2020]. Since
the tables are concise you can do the same with these.

The easiest way
to check the correctness of probability grids (i.e., *.pkl files) is to
read them into Pandas and subtract from the original grids (i.e.,
`gsd_prob_grid.pkl` and `qnormal_prob_grid.pkl`). The result should be
a DataFrame with all cells equal or nearly equal to zero. The code snippet
below shows how to load into Pandas one reproduced probability grid and
compare it with the one provided in this repository.
```python
import pandas as pd

repro_gsd_grid = pd.read_pickle("reproduced_gsd_prob_grid.pkl")
gsd_grid = pd.read_pickle("gsd_prob_grid.pkl")
diff_df = gsd_grid - repro_gsd_grid
diff_df.sum().sum()  # Output: -4.6074255521944e-15
```

The reproduced G-test results (i.e., the
`G_test_on_subjective_quality_datasets_chunk000_of_001.csv` file) can
be checked for correctness in the similar vain to probability grids. The most
important column to check is the one called "p-value_gsd". Values in this
column should be compared with those
contained in the [G_test_results.csv](G_test_results.csv) file.
Importantly, please do not expect to observe identical results in both
files. This is beacuse we use the bootstrapped version of the G-test.
For each sample analysed we generate 10,000 random bootstrap samples.
Hence, the final _p_-value changes when you repeat the G-test for the
same stimulus multiple times. If you want to lear more about this procedure
please refer to the
[In-depth Tutorial about Generating *p*-Value P–P Plots for Your Subjective Data](#in-depth-tutorial-about-generating-p-value-pp-plots-for-your-subjective-data)
section.

## Run G-test on Random Stimuli
If you do not have time to reproduce all G-test results on a single machine
or do not have access to a computational cluster, you can always use the
`reproduce.py` script to run the G-test for _N_ random stimuli. To do so
please run the following in the terminal
```bash
$ python3 reproduce.py -n 3 4
```
Called this way, `reproduce.py` will choose three random stimuli and run
the G-test for them. When done, it will print the estimated parameters
of the GSD and related G-test _p_-values. It will also store the results
in the `g_test_res_for_random_stimuli.csv` file. Based on stimuli IDs
and experiment IDs to which they belong, you can compare the reproduced
results with those in the [G_test_results.csv](G_test_results.csv) file.

The listing below shows an exemplary output of this scenario.
```text
Summary of G-test results:
              psi_hat rho_hat p_value
(2111.0, 2.0)     1.5   0.875  0.5846
Stored the G-test results in the g_test_res_for_random_stimuli.csv file
```
In this case, the result concerns stimulus with ID 2111 that comes from
experiment with ID 2.

# Authorship

The code in this repo was mostly written by Jakub Nawała <jnawala@agh.edu.pl>. Some smaller pieces of it were
implemented by Krzysztof Rusek <krusek@agh.edu.pl>.

Lucjan Janowski <janowski@kt.agh.edu.pl> governed the whole process and Bogdan Ćmiel <cmielbog@gmail.com> made sure
we got statistics right.

# Bibliography
[Janowski2019] Janowski, L., Ćmiel, B., Rusek, K., Nawała, J., & Li, Z. (2019). Generalized Score Distribution.
Retrieved from http://arxiv.org/abs/1909.04369

[Pinson2010] Pinson, M., Speranza, F., Takahashi, A., Schmidmer, C., Lee, C., Okamoto, J., … Dhondt, Y. (2010). Report
on the Validation of Video Quality Models for High Definition Video Content (HDTV Phase I). Retrieved from
https://www.its.bldrdoc.gov/vqeg/projects/hdtv/hdtv.aspx

[Nawała2020] Nawała, J., Janowski, L., Ćmiel, B., & Rusek, K. (2020). Describing Subjective Experiment Consistency by
p-Value P–P Plot. Proceedings of the 28th ACM International Conference on Multimedia (MM ’20).
