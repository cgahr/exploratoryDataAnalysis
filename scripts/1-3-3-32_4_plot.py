# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from exploratory_data_analysis.plots import four_plot, lag_plot, normal_probability_plot

# %%
sns.set_context("notebook")
sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# # Data Initialization

# %%
normal = np.random.normal(0, 1, 20)
linear = np.linspace(0, 1, 20) + np.random.normal(0, 0.1, 20)

# %% [markdown]
# # Run Sequence Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/runseqpl.htm
#
# Basically data per index:

# %%
_ = plt.plot(normal)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")
_ = plt.xlabel("x")
_ = plt.ylabel("y")

# %%
_ = plt.plot(linear)
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
lag_plot(normal)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")

# %%
lag_plot(linear)
_ = plt.title(r"$y = 0.05x + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.1)$")

# %% [markdown]
# # Histogram
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/histogra.htm

# %%
_ = plt.hist(normal)
_ = plt.title(r"$y \sim \mathcal{N}(0, 1)$")
_ = plt.ylabel(r"count")

# %%
_ = plt.hist(linear)
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
normal_probability_plot(normal, alpha=0.05, plot_ci=True)
_ = plt.title(r"$Y \sim \mathcal{N}(0, 1)$")

# %%
# x = np.linspace(0, 1, 100) + np.random.normal(0, 0.01, 100)
normal_probability_plot(linear, alpha=0.05, plot_ci=True)
_ = plt.title(r"$Y \sim \mathcal{U}(0, 1)$")

# %% [markdown]
# # 4-Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/4plot.htm

# %%
four_plot(normal)
plt.suptitle(r"$y \sim \mathcal{N}(0, 1)$")
plt.tight_layout()

# %%
four_plot(linear)
plt.suptitle(r"$Y = 0.05X + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, 0.1)$")
plt.tight_layout()
