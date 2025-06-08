# SPDX-FileCopyrightText: 2023-present University of Vienna
# SPDX-FileCopyrightText: 2023-present Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino
# SPDX-License-Identifier: GPL-3.0-or-later

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

from ._config import (
    CUDA_VERSION,
    DTYPE,
    IS_CPP_ENABLED,
    IS_CUDA_ENABLED,
    IS_SINGLE_PRECISION,
)

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

from . import lags, mask, noise_est, weights, window
from ._ddm import ddm
from ._io import load
from .azimuthalaverage import azimuthal_average, azimuthal_average_array
from .utils import images2numpy, read_images, tiff2numpy

__all__ = [
    "ddm",
    "lags",
    "mask",
    "noise_est",
    "weights",
    "window",
    "load",
    "azimuthal_average",
    "azimuthal_average_array",
    "images2numpy",
    "read_images",
    "tiff2numpy",
    "IS_CPP_ENABLED",
    "IS_CUDA_ENABLED",
    "IS_SINGLE_PRECISION",
    "CUDA_VERSION",
    "DTYPE",
]
