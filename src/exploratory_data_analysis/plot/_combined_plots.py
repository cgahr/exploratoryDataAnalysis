from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.pyplot import Axes

from .._utils import export, flatten_or_raise
from ._regression import normal_probability_plot


@export
def four_plot(y: npt.ArrayLike):
    y = flatten_or_raise(y)
    _, ax = plt.subplots(2, 2)

    ax_ = ax[0][0]
    ax_.plot(y)
    ax_.set_xlabel(r"Index")
    ax_.set_ylabel(r"$Y_i$")

    ax_ = ax[0][1]
    ax_.scatter(y[:-1], y[1:])
    ax_.set_xlabel(r"$Y_{i-1}$")
    ax_.set_ylabel(r"$Y_i$")

    ax_ = ax[1][0]
    ax_.hist(y)
    ax_.set_xlabel("Y")
    ax_.set_ylabel("count")

    ax_ = ax[1][1]
    normal_probability_plot(y, plot_ci=False, alpha=0.05)

    plt.tight_layout()


@export
def bootstrap_mmm_plot(x: npt.ArrayLike, n_samples: int = 500):
    x = flatten_or_raise(x)

    sample = np.random.choice(x, (n_samples, x.shape[0]))

    mean = sample.mean(axis=1)
    median = np.median(sample, axis=1)
    midrange = 0.5 * (sample.max(axis=1) + sample.min(axis=1))

    _, bins = np.histogram(np.concatenate((mean, median, midrange)))

    def _plot(
        y: npt.NDArray[Any], n: int, name: str, ax0: Axes, ax1: Axes, value: float
    ):
        c025 = np.percentile(y, 2.5)
        c975 = np.percentile(y, 97.5)

        _ = ax0.plot(y, range(n), ".")
        _ = ax0.vlines(c025, 0, n, "k")
        _ = ax0.vlines(c975, 0, n, "k")

        _ = ax1.hist(y, bins=bins)
        _ = ax1.vlines(c025, 0, n, "k")
        _ = ax1.vlines(c975, 0, n, "k")

        _ = ax0.set_title(f"{name}={value:.3}\n[{c025:.3}, {c975:.3}]")

    fig, ax = plt.subplots(2, 3, sharex=True, sharey="row")

    _plot(mean, n_samples, "Mean", ax[0][0], ax[1][0], x.mean())
    _plot(median, n_samples, "Median", ax[0][1], ax[1][1], np.median(x))
    _plot(
        midrange, n_samples, "Midrange", ax[0][2], ax[1][2], 0.5 * (x.min() + x.max())
    )

    _ = ax[0][0].set_ylabel("Subsample")
    _ = ax[1][0].set_ylabel("Count")

    fig.suptitle("Bootstrap Plot")

    plt.tight_layout()
