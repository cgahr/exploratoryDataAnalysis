from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy


def confidence_interval_scaling(alpha: float, n_samples: int) -> np.ndarray:
    return np.abs(scipy.stats.t.ppf(alpha / 2, n_samples - 2))


def get_ax(ax: Optional[plt.Axes] = None) -> plt.Axes:
    return plt.gca() if ax is None else ax


def flatten_or_raise(x: npt.ArrayLike) -> np.ndarray:
    x = np.array(x)
    shape: tuple[int, ...] = x.shape
    if shape.count(1) >= x.ndim - 1:
        return x.flatten()

    raise ValueError("The input 'x' cannot be flattened to 1 dim array.")
