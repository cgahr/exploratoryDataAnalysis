import matplotlib.pyplot as plt
import numpy as np
import scipy


def confidence_interval_scaling(alpha, n_samples):
    return np.abs(scipy.stats.t.ppf(alpha / 2, n_samples - 2))


def get_ax(ax):
    if ax is None:
        ax = plt.gca()
    return ax
