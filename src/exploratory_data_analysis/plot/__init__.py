import seaborn as _sns

from . import _config
from ._autocorrelation import *
from ._boxcox import *
from ._combined_plots import *
from ._regression import *
from ._simple_plots import *

_sns.set_context("notebook")
_sns.set_style("darkgrid")
_sns.set_palette(_config.COLORS)
