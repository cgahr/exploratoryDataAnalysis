from collections import namedtuple
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy

from src import utils
from src.utils import get_ax


def _bootstrap_regression(x: np.ndarray, result: Any, alpha: float):
    ci_scaling = utils.confidence_interval_scaling(alpha, len(x))
    bootstrap_intercept = np.random.normal(
        result.intercept,
        ci_scaling * result.intercept_stderr,
        (len(x), 500),
    )
    bootstrap_slope = np.random.normal(
        result.slope,
        ci_scaling * result.stderr,
        (len(x), 500),
    )
    bootstrap = bootstrap_slope * np.tile(x, (500, 1)).T + bootstrap_intercept

    return (
        np.percentile(bootstrap, alpha * 50, axis=1),
        np.percentile(bootstrap, 100 - alpha * 50, axis=1),
    )


def regression_plot(
    x: npt.ArrayLike,
    y: Optional[npt.ArrayLike] = None,
    *,
    plot_ci: bool = True,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
):
    if y is None:
        y = np.array(x)
        x = np.arange(len(y))
    else:
        x = np.array(x)
        y = np.array(y)

    regression = scipy.stats.linregress(x, y)

    y_fit = regression.slope * x + regression.intercept

    ci_scaling = utils.confidence_interval_scaling(alpha, len(x))

    ax = get_ax(ax)
    ax.scatter(x, y)
    ax.plot(x, y_fit, color="black")

    if plot_ci:
        _lb, _ub = _bootstrap_regression(x, regression, alpha)
        ax.plot(x, _lb, "--", color="black")
        ax.plot(x, _ub, "--", color="black")

    intercept_err = ci_scaling * regression.intercept_stderr
    slope_err = ci_scaling * regression.stderr

    ret = {
        "slope": regression.slope,
        "intercept": regression.intercept,
        "slope_err": slope_err,
        "intercept_err": intercept_err,
    }
    return namedtuple("regression", ret)(**ret)


def normal_probability_plot(
    y: npt.ArrayLike,
    *,
    plot_ci: bool = True,
    alpha: float = 0.05,
    ax: Optional[plt.Axes] = None,
):
    y = np.array(y)

    osm, osr = scipy.stats.probplot(y, fit=False)

    ax = get_ax(ax)

    slope, intercept, slope_err, intercept_err = regression_plot(
        osm, osr, plot_ci=plot_ci, alpha=alpha, ax=ax
    )
    ax.set_xlabel(
        fr"$\mathcal{{N}}({intercept:.3f} \pm {intercept_err:0.3f}, "
        + fr"{slope:.3f} \pm {slope_err:.3f})$"
    )
    ax.set_ylabel(r"$Y$")
