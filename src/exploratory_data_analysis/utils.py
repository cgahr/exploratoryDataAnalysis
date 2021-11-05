import sys
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy


def confidence_interval_scaling(alpha: float, n_samples: int) -> float:
    ret: float = scipy.stats.t.ppf(alpha / 2, n_samples - 2)
    return abs(ret)


def get_ax(ax: Optional[plt.Axes] = None) -> plt.Axes:
    return plt.gca() if ax is None else ax


def flatten_or_raise(x: npt.ArrayLike) -> npt.NDArray[Any]:
    x = np.array(x)
    # shape: tuple[int, ...] = x.shape
    if x.shape.count(1) >= x.ndim - 1:
        return x.flatten()

    raise ValueError("The input 'x' cannot be flattened to 1 dim array.")


def export(fun: Callable[..., Any]):
    mod = sys.modules[fun.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(fun.__name__)  # type: ignore
    else:
        mod.__all__ = [fun.__name__]  # type: ignore
    return fun
