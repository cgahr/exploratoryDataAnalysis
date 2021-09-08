# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Exploratory Data Analysis
#     language: python
#     name: eda
# ---

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

# %%
sns.set_context("notebook")
sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

# %%


def autocorrelation_plot(data, *, lag=None, stem=True, ax=None, **kwargs):
    n = len(data)

    if ax is None:
        ax = plt.gca()

    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / n

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / n / c0

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
    ax.axhline(y=z95 / np.sqrt(n), color="grey")
    ax.axhline(y=0.0, color="black")
    ax.axhline(y=-z95 / np.sqrt(n), color="grey")
    ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")

    if lag is None:
        x = np.arange(n) + 1
    else:
        x = np.arange(lag) + 1

    y = [r(loc) for loc in x]

    if stem:
        _, _, baseline = ax.stem(x, y, **kwargs)
        baseline.set_color("none")
    else:
        ax.plot(x, y, **kwargs)

    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    # ax.set_ylim((-1.1, 1.1))


# %% [markdown]
# # Autocorrelation Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda331.htm
#
#

# %% [markdown]
# ## Normal Distribution
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3311.htm
#
# - there are no significant autocorrelations
# - there are very few measurements outside of the confidence interval, however, for the
# 95\% confidence interval, we expect around 1 in 20  to lie outside

# %%
x = np.random.normal(0, 1, 100)

autocorrelation_plot(x)
plt.gca().set_title(r"$Y \sim \mathcal{N}(0, 1)$")


# %% [markdown]
# ## Flicker Dataset
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3312.htm
#
# - the data comes from an autoregressive model with moderate autocorrelation
#
# Thus we should model the data as
# $$Y_i = A_0 + A_1 \cdot Y_{i-1} + E.$$

# %%
x = pd.read_csv("../datasets/flicker.csv")["y"].values

fig, ax = plt.subplots(1, 2)

autocorrelation_plot(x, stem=False, ax=ax[0])
autocorrelation_plot(x, stem=True, lag=80, ax=ax[1])
fig.suptitle(r"flicker dataset")

plt.tight_layout()

# %% [markdown]
# ## Random Walk Dataset
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3313.htm
#
# - the data comes from an autoregressive model with moderate autocorrelation in
# positive direction
#
# Thus we should model the data as
# $$Y_i = A_0 + A_1 \cdot Y_{i-1} + E.$$

# %%
x = pd.read_csv("../datasets/randomwalk.csv")["y"].values

fig, ax = plt.subplots(1, 2)

autocorrelation_plot(x, stem=False, ax=ax[0])
autocorrelation_plot(x, stem=True, lag=125, ax=ax[1])
fig.suptitle(r"random walk dataset")

plt.tight_layout()

# %% [markdown]
# ## Sinusoidal Model / Deflection Dataset
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda3314.htm
#
# - the data comes from an sinusoidal model
#

# %%
x = pd.read_csv("../datasets/deflection.csv")["deflection"].values

fig, ax = plt.subplots(1, 2)

autocorrelation_plot(x, stem=False, ax=ax[0])
autocorrelation_plot(x, lag=50, ax=ax[1])
fig.suptitle(r"sinusoidal model / deflection dataset")

plt.tight_layout()
