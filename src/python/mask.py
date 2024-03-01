# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""This module contains the helper functions for masks.

The functions should be used when computing the azimuthal average of the
image structure function.

.. code-block:: python

    # compute image structure function dqt
    import fastddm as fddm
    ...

    # compute azimuthal average and mask central cross
    mask = fddm.mask.central_cross_mask(dqt.full_shape()[1:])
    aa = fddm.azimuthal_average(dqt,
                                bins=bins,
                                range=(kmin,kmax),
                                mask=mask)
"""

from typing import Optional, Tuple
import numpy as np


def central_cross_mask(
    shape: Tuple[int, int],
    kx: Optional[np.ndarray] = None,
    ky: Optional[np.ndarray] = None
) -> np.ndarray:
    """Evaluate mask to remove central cross from azimuthal average.
    If ``kx`` or ``ky`` are not given, the half-plane representation for the 2D
    image structure function is assumed (i.e., the 0th column and the row at
    ``shape[0] // 2`` are masked out).

    Parameters
    ----------
    shape : Tuple[int, int]
        Shape of the full array, e.g., (128, 256).
    kx : numpy.ndarray, optional
        The array of spatial frequencies along axis `x`. Default is None.
    ky : numpy.ndarray, optional
        The array of spatial frequencies along axis `y`. Default is None.

    Returns
    -------
    numpy.ndarray
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
