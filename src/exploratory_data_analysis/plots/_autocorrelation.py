from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy

from .._utils import export, flatten_or_raise


@export
def autocorrelation_plot(
    y: npt.ArrayLike,
    *,
    maxlag: Optional[int] = None,
    stem: bool = True,
    ax: plt.Axes = None,
    **kwargs
):
    y = flatten_or_raise(y)

    n = len(y)

    if ax is None:
        ax = plt.gca()

    mean = np.mean(y)
    corr0 = np.sum((y - mean) ** 2) / n

    def r(h, y):
        return ((y[: n - h] - mean) * (y[h:] - mean)).sum() / n / corr0

    c95 = np.abs(scipy.stats.t.ppf(0.05 / 2, n - 2))
    c99 = np.abs(scipy.stats.t.ppf(0.01 / 2, n - 2))

    _ = ax.axhline(y=c99, linestyle="--", color="grey")
    _ = ax.axhline(y=c95, color="grey")
    _ = ax.axhline(y=0.0, color="black")
    _ = ax.axhline(y=-c95, color="grey")
    _ = ax.axhline(y=-c99, linestyle="--", color="grey")

    if maxlag is None:
        x = np.arange(n) + 1
    else:
        x = np.arange(maxlag) + 1

    y = [r(loc, y) for loc in x]

    if stem:
        _, _, baseline = ax.stem(x, y, **kwargs)
        baseline.set_color("none")
    else:
        _ = ax.plot(x, y, **kwargs)

    _ = ax.set_xlabel("Lag")
    _ = ax.set_ylabel("Autocorrelation")
    # ax.set_ylim((-1.1, 1.1))
