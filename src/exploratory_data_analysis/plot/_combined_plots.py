from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

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
def bootstrap_plot(
    x: npt.ArrayLike,
    funs: list[Callable[[npt.NDArray[Any]], float]],
    names: list[str],
    n_samples: int = 500,
    alpha: float = 0.05,
):
    x = flatten_or_raise(x)

    sample = np.random.choice(x, (n_samples, len(x)))
    stats = [np.apply_along_axis(fun, 1, sample) for fun in funs]

    _, bins = np.histogram(np.concatenate(stats))
    fig, axes = plt.subplots(2, len(stats), sharex=True, sharey="row")

    for idx, (ax0, ax1) in enumerate(zip(axes[0], axes[1])):
        c025 = np.percentile(stats[idx], 100 * (alpha / 2))
        c975 = np.percentile(stats[idx], 100 * (1 - alpha / 2))

        _ = ax0.plot(stats[idx], range(n_samples), ".")
        _ = ax0.vlines(c025, 0, n_samples, "k")
        _ = ax0.vlines(c975, 0, n_samples, "k")

        _ = ax1.hist(stats[idx], bins=bins, density=True)
        _ = ax1.vlines(c025, 0, n_samples, "k")
        _ = ax1.vlines(c975, 0, n_samples, "k")

        value = funs[idx](x)
        _ = ax0.set_title(f"{names[idx]}={value:.3}\n[{c025:.3}, {c975:.3}]")

    _ = axes[0][0].set_ylabel("Subsample")
    _ = axes[1][0].set_ylabel("Density")

    fig.suptitle("Bootstrap Plot")

    plt.tight_layout()


@export
def bootstrap_mmm_plot(x: npt.ArrayLike, n_samples: int = 500, alpha: float = 0.05):
    bootstrap_plot(
        x,
        [np.mean, np.median, lambda x: (x.max() - x.min()) / 2],
        ["Mean", "Median", "Midrange"],
        n_samples=n_samples,
        alpha=alpha,
    )
