import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_autocorrelation(
    y: npt.ArrayLike,
    *,
    maxlag: int = None,
    stem: bool = True,
    ax: plt.Axes = None,
    **kwargs
):
    y = np.array(y)
    assert len(y.shape) == 1

    n = len(y)

    if ax is None:
        ax = plt.gca()

    mean = np.mean(y)
    corr0 = np.sum((y - mean) ** 2) / n

    def r(h):
        return ((y[: n - h] - mean) * (y[h:] - mean)).sum() / n / corr0

    z95 = 1.959963984540054
    z99 = 2.5758293035489004
    _ = ax.axhline(y=z99 / np.sqrt(n), linestyle="--", color="grey")
    _ = ax.axhline(y=z95 / np.sqrt(n), color="grey")
    _ = ax.axhline(y=0.0, color="black")
    _ = ax.axhline(y=-z95 / np.sqrt(n), color="grey")
    _ = ax.axhline(y=-z99 / np.sqrt(n), linestyle="--", color="grey")

    if maxlag is None:
        x = np.arange(n) + 1
    else:
        x = np.arange(maxlag) + 1

    y = [r(loc) for loc in x]

    if stem:
        _, _, baseline = ax.stem(x, y, **kwargs)
        baseline.set_color("none")
    else:
        _ = ax.plot(x, y, **kwargs)

    _ = ax.set_xlabel("Lag")
    _ = ax.set_ylabel("Autocorrelation")
    # ax.set_ylim((-1.1, 1.1))
