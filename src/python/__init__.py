# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

"""fastddm is the top-level Python package.

It consists of the functions and classes that perform and manage the
Differential Dynamic Microscopy analysis.
"""

# automatically retrieve version
import importlib.metadata
from importlib.metadata import PackageNotFoundError

try:
    __version__ = importlib.metadata.version("fastddm")
except PackageNotFoundError:
    # package is not installed
    pass

from ._config import *

if IS_CUDA_ENABLED:
    import sys

    # On Windows, we need to add the CUDA Toolkit path to the DLL search path
    if sys.platform == "win32":
        import os

        # Get the CUDA Toolkit version
        cuda_version = CUDA_VERSION.split(".")
        cuda_v_major, cuda_v_minor = cuda_version[:2]
        os.add_dll_directory(
            os.path.join(os.environ[f"CUDA_PATH_V{cuda_v_major}_{cuda_v_minor}"], "bin")
        )

from ._ddm import ddm
from .azimuthalaverage import azimuthal_average, azimuthal_average_array
from ._io import load
from .utils import tiff2numpy, images2numpy, read_images
from . import lags, mask, weights, window, noise_est
