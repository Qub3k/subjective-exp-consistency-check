# About

This repository stores data, source codes and an in-depth tutorial all related with
generating *p*-value P&ndash;P plots to assess consistency of subjective data.

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
[p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
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
[p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
file.

[The subjective_quality_datasets.csv file](subjective_quality_datasets.csv) contains subjective
scores from 21 real-life subjective experiments. For more details please refer to [Nawała2020].

[The G_test_results.csv file](G_test_results.csv) contains results of the G-test of goodness-of-fit
(for the fitted GSD distribution) for all stimuli listed in
[the subjective_quality_datasets.csv file](subjective_quality_datasets.csv). To learn what is
the meaning of this file please refer to [Nawała2020] and the
[p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
file.

[The hdtv1_exp1_scores_pp_plot_ready.csv file](hdtv1_exp1_scores_pp_plot_ready.csv) contains
subjective scores from the first experiment of the HDTV Phase I subjective study [Pinson2010].
These scores are used in the
[p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
file to present what input data format is accepted by the `friendly_gsd.py` script.

[The `friendly_gsd.py` script](friendly_gsd.py) depends on several auxiliary scripts. These are:
`gsd.py`, `bootstrap.py`, `G_test_on_real_data.py`, `probability_grid_estimation.py`,
`_logger.py` and `qnormal.py`.

[The *figures* folder](figures) contains graphics used in the
[p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
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
I test. **Please be advised that this may take long time to finish**. Our internal trials on consumer-grade laptop (as
of 2020) indicate that it takes around 5 minutes to process a single stimulus. This is why the script supports
batch processing. More details on this subject are in [the Batch Processing section](#batch-processing).

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
of the [p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
file, respectively.

**WARNING**: Although the code run this way returns G-test results these are not meaningful and
should be discarded. For the explanation why this is the case please refer to the
[p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
file.

# In-depth Tutorial about Generating *p*-Value P–P Plots for Your Subjective Data

If you have your subjective data and would like to generate a *p*-value P–P plot based on its contents then you are in
the right place. This section will walk you through the process. It contains both a high-level description of the
method, as well as code fragments from the [friendly_gsd.py](friendly_gsd.py) script. These two things combined should
give you a quite solid understanding of the methodology. The section also describes issues related with using
the code in this repository&mdash;how to run it, what input does it require, what output it produces and how to use its
batch processing functionality. 

In order not to make this README too lengthy we provide the tutorial in the form of a Jupyter Notebook.
Please open the [p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb)
file to get access to the tutorial. Although GitHub is able to nicely render the file in your browser some images
are not displayed correctly. Please install the `notebook` PIP package and run the following (from repo's root folder)
to open the tutorial locally.
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
of the [p_value_pp_plot_in-depth_tutorial.ipynb](p_value_pp_plot_in-depth_tutorial.ipynb) file.

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
