# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Fabian Krautgasser
# Maintainer: Fabian Krautgasser

"""The collection of python functions to perform Differential Dynamic Microscopy."""

from typing import Optional, Tuple, Callable, Dict

import numpy as np
import scipy.fft as scifft

from ._config import DTYPE


# computational stuff #############################################################################
def autocorrelation(spatial_fft: np.ndarray, *, workers: int = 2) -> np.ndarray:
    """Calculate the autocorrelation function according to the Wiener-Khinchin theorem.

    If the input `spatial_fft` is a power of two, scipy's FFT algorithm is more efficient. The
    spatial input array is also zero-padded for the FFT in time, and then cropped back to the
    original size, i.e. from input (N, y, x) -> (2N, y, x) for the FFT in time, then -> (N, y, x)
    in the returned array.

    Parameters
    ----------
    spatial_fft : np.ndarray
        The spatial FFT of a sequence of images.
    workers : int, optional
        The number of workers (threads) to be passed to scipy.fft, by default 2

    Returns
    -------
    np.ndarray
        The (real) autocorrelation function with the same shape as the input array.
    """
    # fft in time with zero padding of input data
    fft = scifft.fft(
        spatial_fft.astype(np.complex128),
        n=len(spatial_fft) * 2,
        axis=0,
        workers=workers,
    )
    powerspec = np.abs(fft) ** 2
    ifft = scifft.ifft(powerspec, axis=0, workers=workers)

    # returning the real part, cropped to original input length
    return ifft[: len(spatial_fft)].real.astype(DTYPE)


def _diff_image_structure_function(
    rfft2: np.ndarray,
    rfft2_square_mod: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Calculate the image structure function the 'diff' way.

    Uses the rfft2 and square modulus of the rfft2.

    Parameters
    ----------
    rfft2 : np.ndarray
        The rfft2 of the image series.
    rfft2_square_mod : np.ndarray
        Square modulus of the above input.
    lag : int
        The lag time in number of frames.

    Returns
    -------
    np.ndarray
        The half plane image structure function.

    Raises
    ------
    RuntimeError
        If array of lags is longer than image series.
    """
    length, *_ = rfft2.shape

    if lag >= length:
        raise RuntimeError("Time delay cannot be longer than the timeseries itself!")

    # use slice objects to handle cropping of arrays better
    crop = slice(None, -lag) if lag != 0 else slice(None, None)
    shift = slice(lag, None)

    cropped_conj = rfft2[crop].conj()
    shifted = rfft2[shift]
    shifted_abs_square = rfft2_square_mod[shift]
    cropped_abs_square = rfft2_square_mod[crop]

    sum_of_parts = (
        shifted_abs_square + cropped_abs_square - 2 * (cropped_conj * shifted).real
    )
    dqt = np.mean(sum_of_parts, axis=0)

    return dqt


def image_structure_function(
    sq_mod_cumsum: np.ndarray,
    sq_mod_cumsum_rev: np.ndarray,
    autocorrelation: np.ndarray,
    lag: int,
) -> np.ndarray:
    """Calculate the image structure function.

    Uses the square modulus of a (half-plane) spatial FFT timeseries and its autocorrelation
    function.

    Taken from Eq. (2) from 10.1140/epje/s10189-021-00146-2. Adjusted to use the autocorrelation
    function computed via the wiener-khinchin theorem.

    Returns the average over the whole timeseries (average over the 0th axis).

    Parameters
    ----------
    sq_mod_cumsum : np.ndarray
        The cumulative sum of the square modulus of the spatial FFT of an image time series, e.g.
        np.cumsum(np.abs(rfft2(imgs))**2, axis=0).
    sq_mod_cumsum_rev : np.ndarray
        The cumulative sum of the reversed square modulus of the spatial FFT of an image time
        series, e.g. np.cumsum(np.abs(rfft2(imgs)[::-1])**2, axis=0).
    autocorrelation : np.ndarray
        The autocorrelation function of the spatial FFT of an image timeseries.
    lag : int
        The delay time in frame units, must be 0 <= lag <= len(square_modulus)

    Returns
    -------
    np.ndarray
        The half-plane image structure function.

    Raises
    ------
    RuntimeError
        If the given lag time is longer than the timeseries.
    """
    length, *_ = sq_mod_cumsum.shape

    if lag >= length:
        raise RuntimeError("Time lag cannot be longer than the timeseries itself!")

    autocorrelation = autocorrelation[lag].real

    offset = lag + 1
    sum_of_parts = (
        sq_mod_cumsum[-offset] + sq_mod_cumsum_rev[-offset] - 2 * autocorrelation
    )
    sum_of_parts /= length - lag  # normalization

    return sum_of_parts  # half plane


def _py_image_structure_function(
    images: np.ndarray,
    lags: np.ndarray,
    nx: int,
    ny: int,
    window: np.ndarray,
    *,
    mode: str = "fft",
    workers: int = 2,
    **kwargs,
) -> np.ndarray:
    """The handler function for the python image structure function backend.

    Parameters
    ----------
    images : np.ndarray
        Input image series.
    lags : np.ndarray
        Array of lag times.
    nx : int
        The number of Fourier nodes in x direction (for normalization).
    ny : int
        The number of Fourier nodes in y direction (for normalization).
    window : np.ndarray
        A 2D array containing the window function to be applied to the images.
        If window is empty, no window is applied.
    mode : str, optional
        Calculate the autocorrelation function with Wiener-Khinchin theorem ('fft') or classically ('diff'), by default "fft"
    workers : int, optional
        Number of workers to be used by scipy.fft, by default 2

    Returns
    -------
    np.ndarray
        The half-plane image structure function for all given lag times.

    Raises
    ------
    RuntimeError
        If a mode other than the 2 possible ones is given.
    """
    # backend
    backend: Dict[str, Callable] = {
        "diff": _diff_image_structure_function,
        "fft": image_structure_function,
    }

    # sanity check
    modes = list(backend.keys())
    if mode not in modes:
        raise RuntimeError(
            f"Unknown mode '{mode}' for image structure function. Only possible options are "
            f"{modes}."
        )

    # setup
    calc_dqt = backend[mode]  # select function
    length = len(lags)
    dqt = np.zeros(
        (length + 2, ny, nx // 2 + 1),
        dtype=DTYPE,
    )  # +2 for (avg) power spectrum & variance

    # spatial fft & square modulus
    rfft2 = normalized_rfft2(images, nx, ny, window=window, workers=workers)
    square_mod = np.abs(rfft2) ** 2

    if mode == "diff":
        # just needs argument setup
        args = (rfft2, square_mod)

    else:
        # autocorrelation for fft mode
        autocorr = autocorrelation(rfft2, workers=workers)
        cumsum = np.cumsum(square_mod.astype(np.float64), axis=0).astype(DTYPE)
        cumsum_rev = np.cumsum(square_mod[::-1].astype(np.float64), axis=0).astype(
            DTYPE
        )

        args = (cumsum, cumsum_rev, autocorr)

    for i, lag in enumerate(lags):
        dqt[i] = calc_dqt(*args, lag)

    # add the power spectrum and the variance as the last two entries in the dqt array
    dqt[-2] = square_mod.mean(axis=0)
    dqt[-1] = rfft2.var(axis=0)

    return scifft.fftshift(dqt, axes=-2)  # only shift in y


# convenience #####################################################################################
def normalized_rfft2(
    images: np.ndarray,
    nx: int,
    ny: int,
    window: np.ndarray,
    *,
    workers: int = 2,
) -> np.ndarray:
    """Calculate the normalized rfft2.

    The normalization is the square root of the product of the last 2 dimensions of the shape of
    the `images` array. If `nx` *and* `ny` are given, the normalization is the square root of the
    product of `nx` and `ny`, and the input to rfft2 is zero padded to match (ny, nx).

    Parameters
    ----------
    images : np.ndarray
        An image sequence.
    nx : int
        The number of Fourier nodes in x direction.
    ny : int
        The number of Fourier nodes in y direction.
    window : np.ndarray
        A 2D array containing the window function to be applied to the images.
        If window is empty, no window is applied.
    workers : int, optional
        The number of threads to be passed to scipy.fft, by default 2.

    Returns
    -------
    np.ndarray
        The normalized half-plane spatial fft of the image sequence.
    """
    if nx is None or ny is None:
        *_, ny, nx = images.shape

    if len(window) > 0:
        rfft2 = scifft.rfft2(images.astype(DTYPE) * window, s=(ny, nx), workers=workers)
    else:
        rfft2 = scifft.rfft2(images.astype(DTYPE), s=(ny, nx), workers=workers)
    norm = np.sqrt(nx * ny)
    return rfft2 / norm
