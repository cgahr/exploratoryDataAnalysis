# %%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from exploratory_data_analysis.plots import boxcox_nonlinearity_plot

# %%
sns.set_context("notebook")
sns.set_style("darkgrid")

plt.rcParams["figure.figsize"] = (6, 4)
plt.rcParams["figure.dpi"] = 120

# np.random.seed(42)


# %% [markdown]
# # Box-Cox Linearity Plot
#
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda334.htm
#
# The Box-Cox Linearity plot assert if a non-linear transform of the index improves the
# linear fit of the data to the index. The box-cox transformation transforms the index
# via the following transform
#
# $$ T(X) = \begin{cases} (X^\lambda - 1) / \lambda, &\text{if } \lambda \neq 0, \\
# \mathrm{ln}(X), &\text{else}. \end{cases} $$
#
# The box-cox linearity plot shows the correlation coefficient of the transformed input
# $T(X)$ and the variable $Y$ depending on the transformation parameter $\lambda$. The
# optimal $\lambda$ maximizes the absolute correlation.

# %%
y = np.linspace(1, 10, 20) ** 2 + np.random.normal(0, 0.2, 20)
bounds = (-5, 5)

boxcox_nonlinearity_plot(y, bounds=bounds)

# %%
