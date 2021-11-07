# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from exploratory_data_analysis.plots import autocorrelation_plot

# %%
sns.set_context("notebook")
sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

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
_ = plt.gca().set_title(r"$Y \sim \mathcal{N}(0, 1)$")


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
autocorrelation_plot(x, stem=True, maxlag=80, ax=ax[1])
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
autocorrelation_plot(x, stem=True, maxlag=125, ax=ax[1])
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
autocorrelation_plot(x, maxlag=50, ax=ax[1])
fig.suptitle(r"sinusoidal model / deflection dataset")

plt.tight_layout()
