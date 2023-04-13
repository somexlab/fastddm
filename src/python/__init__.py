# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

# automatically retrieve version
from pkg_resources import get_distribution, DistributionNotFound
try:
    __version__ = get_distribution("fastddm").version
except DistributionNotFound:
    # package is not installed
    pass

from ._config import *
from ._ddm import ddm
from .azimuthalaverage import azimuthal_average
from ._io import load
from ._utils import tiff2numpy, images2numpy, read_images
from . import lags, mask, weights, window
