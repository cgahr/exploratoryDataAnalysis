import matplotlib.pyplot as plt
import numpy.typing as npt

from .regression import normal_probability_plot


def four_plot(y: npt.ArrayLike):
    fig, ax = plt.subplots(2, 2)

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
