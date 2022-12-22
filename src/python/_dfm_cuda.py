from typing import List
import numpy as np

from ._memchk import get_free_mem
from ._gpumemchk import get_free_gpu_mem
# from .core_cuda import chk_host_mem_direct, chk_host_mem_fft
from .core_cuda import get_device_pitch, get_device_fft2_mem, get_device_fft_mem
from .core_cuda import dfm_direct_cuda, dfm_fft_cuda

def dfm_direct_gpu(img_seq: np.ndarray, lags: List[int], nx: int, ny: int) -> np.ndarray:
    """Digital Fourier Microscopy, direct mode on GPU

    Compute the image structure function using differences on the GPU.

    Parameters
    ----------
    img_seq : numpy.ndarray
        Input image sequence.
    lags : array_like
        List of selected lags.
    nx : int
        Number of fft nodes in x direction.
    ny : int
        Number of fft nodes in y direction.

    Returns
    -------
    isf : numpy.ndarray
        Image structure function.

    Raises
    ------
    RuntimeError
        If memory is not sufficient to perform the calculations.
    """

    # +++ ANALYZE +++
    return dfm_direct_cuda(img_seq, lags, nx, ny)

def dfm_fft_gpu(img_seq: np.ndarray, lags: List[int], nx: int, ny: int, nt: int) -> np.ndarray:
    """Digital Fourier Microscopy, fft mode on GPU

    Compute the image structure function using the Wiener-Khinchin theorem on
    the GPU.

    Parameters
    ----------
    img_seq : numpy.ndarray
        Input image sequence.
    lags : array_like
        List of selected lags.
    nx : int
        Number of fft nodes in x direction.
    ny : int
        Number of fft nodes in y direction.
    nt : int
        Number of fft nodes in t direction.

    Returns
    -------
    isf : numpy.ndarray
        Image structure function.

    Raises
    ------
    RuntimeError
        If memory is not sufficient to perform the calculations.
    """

    # +++ ANALYZE +++
    return dfm_fft_cuda(img_seq, lags, nx, ny, nt)
