# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""Collection of helper functions for masks of azimuthal average"""

from typing import Optional, Tuple
import numpy as np


def central_cross_mask(
    shape: Tuple[int, int],
    kx: Optional[np.ndarray] = None,
    ky: Optional[np.ndarray] = None
) -> np.ndarray:
    """Evaluate mask to remove central cross from azimuthal average.
    If `kx` or `ky` are not given, the half-plane representation for the 2D
    image structure function is assumed (0th column and row at `shape[0] // 2`
    are masked out).

    Parameters
    ----------
    shape : (int, int)
        Shape of the full array, e.g., (128, 256).
    kx : np.ndarray, optional
        The array of spatial frequencies along axis x. Default is None.
    ky : np.ndarray, optional
        The array of spatial frequencies along axis y. Default is None.

    Returns
    -------
    mask : np.ndarray
        The mask.
    """
    if kx is None or ky is None:
        mask = np.full(shape, True)
        mask[:, 0] = False
        mask[shape[0] // 2] = False

        return mask
    else:
        X, Y = np.meshgrid(kx, ky)
        mask = np.full(shape, True)
        mask[(X == 0.0) | (Y == 0.0)] = False

        return mask
