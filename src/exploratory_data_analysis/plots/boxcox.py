import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy

from .. import utils


def boxcox_nonlinearity_plot(
    y: npt.ArrayLike, *, bounds: tuple[float, float] = (-5.0, 5.0)
):
    y = utils.flatten_or_raise(y)

    _, ax = plt.subplots(2, 2, sharey="row")

    x = np.arange(y.shape[0]) + 1
    slope, intercept = np.polyfit(x, y, deg=1)
    y_fit = slope * x + intercept

    ax[0][0].scatter(x, y)
    ax[0][0].plot(x, y_fit, "k")

    ax[0][0].set_xlabel(f"Residual Std = {(y_fit - y).std():.2f}")
    ax[0][0].set_ylabel("$y$")
    ax[0][0].set_title("Linear Fit of Original Data")

    def boxcox_corr(t):
        return np.corrcoef(scipy.stats.boxcox(x, t), y)[0, 1]

    lmbda = scipy.optimize.minimize_scalar(
        lambda t: -np.abs(boxcox_corr(t)), bounds=bounds
    ).x
    lmbdas = np.linspace(*bounds, 20)
    corr = list(map(boxcox_corr, lmbdas))

    ax[1][0].plot(lmbdas, corr)

    ax[1][0].set_xlabel(fr"$\lambda={lmbda:.3f}$")
    ax[1][0].set_ylabel("correlation")
    ax[1][0].set_title("Box-Cox Linearity Plot")

    x_trans = scipy.stats.boxcox(x, lmbda)
    slope, intercept = np.polyfit(x_trans, y, deg=1)
    y_fit2 = slope * x_trans + intercept

    ax[0][1].scatter(x_trans, y)
    ax[0][1].plot(x_trans, y_fit2, "k")

    ax[0][1].set_xlabel(f"Residual Std = {(y_fit2 - y).std():.2f}")
    ax[0][1].set_title("Linear Fit of Transformed Data")

    ax[1][1].axis("off")

    plt.tight_layout()
