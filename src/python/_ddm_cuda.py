from typing import List, Optional
import numpy as np

# from ._memchk import get_free_mem
# from ._gpumemchk import get_free_gpu_mem
# from .core import chk_host_mem_direct, chk_host_mem_fft
# from .core import get_device_pitch, get_device_fft2_mem, get_device_fft_mem
from ._core import set_device
from ._core import ddm_diff_cuda, ddm_fft_cuda, ddm_diff_cuda_single, ddm_fft_cuda_single

def ddm_diff_gpu(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    gpu_id: Optional[int] = 0
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
    return ddm_diff_cuda(img_seq, lags, nx, ny)

def ddm_diff_gpu_single(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    gpu_id: Optional[int] = 0
) -> np.ndarray:
    """Differential Dynamic Microscopy, diff mode on GPU (single precision)

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
    return ddm_diff_cuda_single(img_seq, lags, nx, ny)

def ddm_fft_gpu(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    nt: int,
    gpu_id: Optional[int] = 0
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
    return ddm_fft_cuda(img_seq, lags, nx, ny, nt)

def ddm_fft_gpu_single(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    nt: int,
    gpu_id: Optional[int] = 0
) -> np.ndarray:
    """Differential Dynamic Microscopy, fft mode on GPU (single precision).

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
    return ddm_fft_cuda_single(img_seq, lags, nx, ny, nt)
