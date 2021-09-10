from typing import Optional

import matplotlib.pyplot as plt
import numpy.typing as npt

from ..utils import get_ax


def lag_plot(x: npt.ArrayLike, *, ax: Optional[plt.Axes] = None):
    ax = get_ax(ax)

    _ = ax.scatter(x[:-1], x[1:])
    _ = ax.set_xlabel(r"$X_{i-1}$")
    _ = ax.set_ylabel(r"$X_i$")
