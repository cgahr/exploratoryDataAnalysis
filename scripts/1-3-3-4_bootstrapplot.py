# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from exploratory_data_analysis.plots import bootstrap_plot

# %%
sns.set_context("notebook")
sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

# %% [markdown]
# # Bootstrap Plot Uniform Data
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda334.htm

# %%
x = np.random.uniform(0, 1, 100)
bootstrap_plot(x)
_ = plt.suptitle(r"Bootstrap $X \sim \mathcal{U}(0, 1)$")

# %% [markdown]
# # Bootstrap Plot Normal Data
#

# %%
x = np.random.normal(0, 1, 100)
bootstrap_plot(x)
_ = plt.suptitle(r"Bootstrap $X \sim \mathcal{N}(0, 1)$")
