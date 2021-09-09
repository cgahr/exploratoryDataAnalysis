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
import scipy
import seaborn as sns

from src.plots import normal_probability_plot

# %%
sns.set_context("notebook")
sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# # Run Sequence Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/runseqpl.htm
#
# Basically data per index:

# %%
x = np.random.normal(0, 1, 20)

_ = plt.plot(x)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")
_ = plt.xlabel("x")
_ = plt.ylabel("y")

# %%
x = np.linspace(0, 1, 20) + np.random.normal(0, 0.1, 20)

_ = plt.plot(x)
_ = plt.title(r"$y = 0.05x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.1)$")
_ = plt.xlabel("x")
_ = plt.ylabel("y")

# %% [markdown]
# # Lag Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/lagplot.htm
#
# Shows if data is correlated or not, plot index i vs index i+1

# %%
x = np.random.normal(0, 1, 20)

_ = plt.scatter(x[:-1], x[1:])
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")
_ = plt.xlabel(r"$X_{i-1}$")
_ = plt.ylabel(r"$X_i$")

# %%
x = np.linspace(0, 1, 20) + np.random.normal(0, 0.1, 20)

_ = plt.scatter(x[:-1], x[1:])
_ = plt.title(r"$y = 0.05x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.1)$")
_ = plt.xlabel(r"$X_{i-1}$")
_ = plt.ylabel(r"$X_i$")

# %% [markdown]
# # Histogram
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm

# %%
x = np.random.normal(0, 1, 20)

_ = plt.hist(x)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")
_ = plt.ylabel(r"count")

# %%
x = np.linspace(0, 1, 20) + np.random.normal(0, 0.1, 20)

_ = plt.hist(x)
_ = plt.title(r"$y = 0.05x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.1)$")
_ = plt.ylabel(r"count")

# %% [markdown]
# # Normal Probability Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/normprpl.htm
#
# The normal probability plot (Chambers et al., 1983) is a graphical technique for
# assessing whether or not a data set is approximately normally distributed.
#
# The data are plotted against a theoretical normal distribution in such a way that the
# points should form an approximate straight line. Departures from this straight line
# indicate departures from normality.

# %%
x = np.random.normal(0, 1, 20)
normal_probability_plot(x, alpha=0.05, plot_ci=True)

_ = plt.title(r"$Y \sim \mathcal{N}(0, 1)$")

# %%
# x = np.linspace(0, 1, 100) + np.random.normal(0, 0.01, 100)
x = np.random.uniform(0, 1, 20)
normal_probability_plot(x, alpha=0.05, plot_ci=True)
_ = plt.title(r"$Y \sim \mathcal{U}(0, 1)$")

# %%

# %% [markdown]
# # 4-Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/4plot.htm

# %%
x = np.random.normal(0, 1, 20)

fig, ax = plt.subplots(2, 2)

ax_ = ax[0][0]
ax_.plot(x)
ax_.set_xlabel(r"$X_i$")
ax_.set_ylabel(r"$Y_i$")

ax_ = ax[0][1]
ax_.scatter(x[:-1], x[1:])
ax_.set_xlabel(r"$Y_{i-1}$")
ax_.set_ylabel(r"$Y_i$")

ax_ = ax[1][0]
ax_.hist(x)
ax_.set_xlabel("Y")
ax_.set_ylabel("count")

ax_ = ax[1][1]
normal_probability_plot(x, plot_ci=True, alpha=0.05)

fig.suptitle(r"$y \sim \mathcal{N}(0, 1)$")
plt.tight_layout()

# %%
x = np.linspace(0, 1, 20) + np.random.normal(0, 0.1, 20)

fig, ax = plt.subplots(2, 2)

ax_ = ax[0][0]
ax_.plot(x)
ax_.set_xlabel(r"$X_i$")
ax_.set_ylabel(r"$Y_i$")

ax_ = ax[0][1]
ax_.scatter(x[:-1], x[1:])
ax_.set_xlabel(r"$Y_{i-1}$")
ax_.set_ylabel(r"$Y_i$")

ax_ = ax[1][0]
ax_.hist(x)
ax_.set_xlabel("Y")
ax_.set_ylabel("count")

ax_ = ax[1][1]
normal_probability_plot(x, plot_ci=True, alpha=0.05)

fig.suptitle(r"$Y = 0.05X + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.1)$")
plt.tight_layout()
