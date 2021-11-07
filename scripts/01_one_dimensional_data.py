# %% [markdown]
# # Showcase: Analyzing One Dimensional Datasets
# This notebook shows the graphical techniques available in this package for analyzing
# one dimensional datasets.

# %% [markdown]
# ## Imports and Definitions

# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import exploratory_data_analysis as eda

# %%
sns.set_context("notebook")
sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

# %%
uniform_500 = np.random.uniform(0, 1, 500)
normal_500 = np.random.normal(0, 1, 500)

normal = np.random.normal(0, 1, 20)
linear = np.linspace(0, 1, 20) + np.random.normal(0, 0.05, 20)

# %% [markdown]
# ## Run Sequence Plot
#
# > Run sequence plots (Chambers 1983) are an easy way to graphically summarize a
# > univariate data set. A common assumption of univariate data sets is that they behave
# > like:
# > 1. random drawings;
# > 1. from a fixed distribution;
# > 1. with a common location; and
# > 1. with a common scale.
# >
# > With run sequence plots, shifts in location and scale are typically quite evident.
# > Also, outliers can easily be detected.
# >
# -- [Engineering Statistics Handbook, Run-Squence Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/runseqpl.htm)
#
# Basically, the run sequence plot plots data vs index.

# %% [markdown]
# ### Run Sequence Plot Normal Data

# %%
_ = plt.plot(normal)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")
_ = plt.xlabel("x")
_ = plt.ylabel("y")

# %% [markdown]
# ### Run Sequence Plot Linear Data

# %%
_ = plt.plot(linear)
_ = plt.title(r"$y = 0.05x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.05)$")
_ = plt.xlabel("x")
_ = plt.ylabel("y")

# %% [markdown]
# ## Lag Plot
#
# >  A lag plot checks whether a data set or time series is random or not. Random data
# > should not exhibit any identifiable structure in the lag plot. Non-random structure
# > in the lag plot indicates that the underlying data are not random.
# >
# -- [Engineering Statistics Handbook, Lag Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm)

# %% [markdown]
# ### Lag Plot Normal Data

# %%
eda.plot.lag_plot(normal)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")

# %% [markdown]
# ### Lag Plot Linear Data

# %%
eda.plot.lag_plot(linear)
_ = plt.title(r"$y = 0.05x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.05)$")

# %% [markdown]
# ## Histogram
#
# > The purpose of a histogram (Chambers) is to graphically summarize the distribution
# > of a univariate data set.
# >
# >The histogram graphically shows the following:
# > 1. center (i.e., the location) of the data;
# > 1. spread (i.e., the scale) of the data;
# > 1. skewness of the data;
# > 1. presence of outliers; and
# > 1. presence of multiple modes in the data.
# >
# -- [Engineering Statistics Handbook, Histogram](https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm)
#

# %% [markdown]
# ### Histogram Normal Data

# %%
_ = plt.hist(normal)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")
_ = plt.ylabel(r"count")

# %% [markdown]
# ### Histogram Linear Data

# %%
_ = plt.hist(linear)
_ = plt.title(r"$y = 0.05x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.05)$")
_ = plt.ylabel(r"count")

# %% [markdown]
# ## Normal Probability Plot
#
# > The normal probability plot (Chambers et al., 1983) is a graphical technique for
# > assessing whether or not a data set is approximately normally distributed.
# >
# > The data are plotted against a theoretical normal distribution in such a way that
# > the points should form an approximate straight line. Departures from this straight
# > line indicate departures from normality.
# >
# -- [Engineering Statistics Handbook, Normal Probability Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm)
#
# In addition, for a given significance level $\alpha$, the plot can show the confidence
# interval of the fitted normal distribution.

# %% [markdown]
# ### Normal Probability Plot Normal Data

# %%
eda.plot.normal_probability_plot(normal, alpha=0.05, plot_ci=True)
_ = plt.title(r"$Y \sim \mathcal{N}(0, 1)$")

# %% [markdown]
# ### Normal Probability Plot Linear Data

# %%
eda.plot.normal_probability_plot(linear, alpha=0.05, plot_ci=True)
_ = plt.title(r"$Y \sim \mathcal{U}(0, 1)$")


# %% [markdown]
# # 4-Plot
#
# >  The 4-plot is a collection of 4 specific EDA graphical techniques whose purpose is
# > to test the assumptions that underlie most measurement processes. A 4-plot consists
# > of a
# > 1. run sequence plot;
# > 1. lag plot;
# > 1. histogram;
# > 1. normal probability plot.
# If the 4 underlying assumptions of a typical measurement process hold, then the above
# > 4 plots will have a characteristic appearance (see the normal random numbers case
# > study below); if any of the underlying assumptions fail to hold, then it will be
# > revealed by an anomalous appearance in one or more of the plots.
# >
# > Although the 4-plot has an obvious use for univariate and time series data, its
# > usefulness extends far beyond that. Many statistical models of the form
# > $$Y_i = f(X_1, \ldots, X_k) + E_i$$
# > have the same underlying assumptions for the error term. That is, no matter how
# > complicated the functional fit, the assumptions on the underlying error term are
# > still the same. The 4-plot can and should be routinely applied to the residuals when
# > fitting models regardless of whether the model is simple or complicated.
# >
# -- [Engineering Statistics Handbook, 4-Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/4plot.htm)
#
# %% [markdown]
# ### 4-Plot Normal Data

# %%
eda.plot.four_plot(normal)
plt.suptitle(r"$y \sim \mathcal{N}(0, 1)$")
plt.tight_layout()

# %% [markdown]
# ### 4-Plot Linear Data
# %%
eda.plot.four_plot(linear)
plt.suptitle(r"$Y = 0.05X + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.05)$")
plt.tight_layout()

# %% [markdown]
# ## Bootstrap Plot
#
# > The bootstrap (Efron and Gong) plot is used to estimate the uncertainty of a
# > statistic.
# >
# -- [Engineering Statistics Handbook, Bootstrap Plot](https://www.itl.nist.gov/div898/handbook/eda/section3/eda334.htm)
#
# The function `eda.plot.bootstrap_mmm_plot' bootstraps the mean, median and midrange of a dataset.

# %% [markdown]
# ### Bootstrap Mean, Median and Midrange of Uniform Data
#

# %%
eda.plot.bootstrap_mmm_plot(uniform_500)
_ = plt.suptitle(r"Bootstrap $X \sim \mathcal{U}(0, 1)$")

# %% [markdown]
# ### Bootstrap Mean, Median and Midrange of Normal Data
#

# %%
eda.plot.bootstrap_mmm_plot(normal_500)
_ = plt.suptitle(r"Bootstrap $X \sim \mathcal{N}(0, 1)$")

# %% [markdown]
# ### Bootstrap Mean, Median and Midrange of Uniform Data
#

# %%
eda.plot.bootstrap_plot(
    uniform_500,
    [np.std, lambda x: np.std(x, ddof=1), lambda x: np.std(x, ddof=2)],
    ["Std", "Std, ddof=1", "Std, ddof=2"],
)
_ = plt.suptitle(r"Bootstrap $X \sim \mathcal{U}(0, 1)$")

# %% [markdown]
# ### Bootstrap Mean, Median and Midrange of Normal Data
#

# %%
eda.plot.bootstrap_plot(
    normal_500,
    [np.std, lambda x: np.std(x, ddof=1), lambda x: np.std(x, ddof=2)],
    ["Std", "Std, ddof=1", "Std, ddof=2"],
)
_ = plt.suptitle(r"Bootstrap $X \sim \mathcal{N}(0, 1)$")
