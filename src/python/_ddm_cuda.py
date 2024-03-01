# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""The collection of CUDA functions to perform Differential Dynamic Microscopy."""

from typing import List, Optional
import numpy as np

from ._core_cuda import set_device
from ._core_cuda import ddm_diff_cuda, ddm_fft_cuda
from ._config import DTYPE


def ddm_diff_gpu(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    window: np.ndarray,
    gpu_id: Optional[int] = 0,
    **kwargs,
) -> np.ndarray:
    """Differential Dynamic Microscopy, diff mode on GPU

    Compute the image structure function using differences on the GPU.

    Parameters
    ----------
    img_seq : numpy.ndarray
        Input image sequence.
    lags : array_like
        List of selected lags.
    nx : int
        Number of fft nodes in x direction
        (for very large transorms, this value must be even).
    ny : int
        Number of fft nodes in y direction.
    window : np.ndarray
        A 2D array containing the window function to be applied to the images.
        If window is empty, no window is applied.
    gpu_id : int, optional
        The ID of the device to be used. Default is 0.

    Returns
    -------
    np.ndarray
        Image structure function.

    Raises
    ------
    RuntimeError
        If memory is not sufficient to perform the calculations
        or if device id is out of bounds.
    """
    # set device
    set_device(gpu_id)

    # analyze
    return ddm_diff_cuda(img_seq, lags, nx, ny, window)


def ddm_fft_gpu(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    nt: int,
    window: np.ndarray,
    gpu_id: Optional[int] = 0,
    **kwargs,
) -> np.ndarray:
    """Differential Dynamic Microscopy, fft mode on GPU.

    Compute the image structure function using the Wiener-Khinchin theorem on
    the GPU.

    Parameters
    ----------
    img_seq : numpy.ndarray
        Input image sequence.
    lags : array_like
        List of selected lags.
    nx : int
        Number of fft nodes in x direction
        (for very large transorms, this value must be even).
    ny : int
        Number of fft nodes in y direction.
    nt : int
        Number of fft nodes in t direction
        (for very large transorms, this value must be even).
    window : np.ndarray
        A 2D array containing the window function to be applied to the images.
        If window is empty, no window is applied.
    gpu_id : int, optional
        The ID of the device to be used. Default is 0.

    Returns
    -------
    np.ndarray
        Image structure function.

    Raises
    ------
    RuntimeError
        If memory is not sufficient to perform the calculations.
    """
    # set device
    set_device(gpu_id)

    # analyze
    return ddm_fft_cuda(img_seq, lags, nx, ny, nt, window)
