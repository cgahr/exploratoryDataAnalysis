from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy


def confidence_interval_scaling(alpha: float, n_samples: int) -> np.ndarray:
    return np.abs(scipy.stats.t.ppf(alpha / 2, n_samples - 2))


def get_ax(ax: Optional[plt.Axes] = None) -> plt.Axes:
    return plt.gca() if ax is None else ax
