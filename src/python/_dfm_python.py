"""The collection of python functions to perform DDM."""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.fft as scifft
import skimage.io as io
import tifffile


def tif_to_numpy(path: Path, seq: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read a TIFF file (or a sequence inside a multipage TIFF) and return it as a numpy array.

    Parameters
    ----------
    path : Path
        The path to the TIFF file.
    seq : Optional[Sequence[int]], optional
        A sequence, e.g. `range(5, 10)`, to describe a specific range within a multipage TIFF, by default None

    Returns
    -------
    np.ndarray
        The array containing the image information; coordinate convention is (z,y,x).
    """
    if seq is None:
        return io.imread(path)

    # load the given image sequence with tifffile
    with tifffile.TiffFile(path) as tif:
        data = tif.asarray(key=seq)
    return data


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
    fft = scifft.fft(spatial_fft, n=len(spatial_fft) * 2, axis=0, workers=workers)
    powerspec = np.abs(fft) ** 2
    ifft = scifft.ifft(powerspec, axis=0, workers=workers)

    # returning the real part, cropped to original input length
    return ifft[: len(spatial_fft)].real


@lru_cache()
def distance_array(
    shape: Tuple[int, ...], r_centre: Optional[int] = None
) -> np.ndarray:
    """Calculate the array of distances for a given radius centre point `r_centre`.

    The last two values of the input shape are used. If the shape is not square, the `r_centre`
    argument is ignored, and a `r_centre` equal to the integer division of each dimension by 2 is
    used.

    If the input shape is square, and no centre is supplied, then `r_centre` is the integer
    division of the square dimension by 2.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape of an array; only the last 2 dimensions are considered.
    r_centre : Optional[int], optional
        The centre point to calculate the distance from, only for square shapes, by default None

    Returns
    -------
    np.ndarray
        The array of distances to the centre point.
    """
    # helper function
    @lru_cache()
    def dist_2d(i, j, r_i, r_j):
        """Calculate the distance of a point (i, j) with respect to another point (r_i, r_j)."""
        return np.sqrt((j - r_j) ** 2 + (i - r_i) ** 2)

    *rest, y, x = shape
    dist = np.zeros((y, x))

    # non-square dimensions
    if x != y:
        r_centre_y, r_centre_x = y // 2, x // 2

        for j in range(y):
            for i in range(x):
                dist[j, i] = dist_2d(i, j, r_centre_x, r_centre_y)
        return dist

    # below for square dimensions
    if r_centre is None:
        r_centre = x // 2

    for j in range(y):
        for i in range(x):
            dist[j, i] = dist_2d(i, j, r_centre, r_centre)

    return dist


def azimuthal_average(
    image: np.ndarray, dist: Optional[np.ndarray] = None
) -> np.ndarray:
    """Calculate the azimuthal average for a given square image.

    Can supply array of distances from the center to avoid calculating it multiple times (however
    it does have the `lru_cache` decorator).

    Parameters
    ----------
    image : np.ndarray
        A 2D square image numpy array.
    dist : Optional[np.ndarray], optional
        An array of distances from a centre point, by default None

    Returns
    -------
    np.ndarray
        The azimuthally averaged image, the length of half the dimension.

    Raises
    ------
    RuntimeError
        If the input images is not square.
    """
    y, x = image.shape

    if x != y:
        raise RuntimeError(f"Dimensions for X ({x}) and Y ({y}) not equal!")

    # setup array of distances to center of array
    r_centre = x // 2
    if dist is None:
        dist = distance_array(image.shape, r_centre)

    # list of radii, basically index count from centre
    radii = np.arange(1, r_centre + 1)
    azimuthal_average = np.zeros_like(radii, dtype=np.float64)

    for i, r in enumerate(radii):
        azimuthal_average[i] = image[(dist >= r - 0.5) & (dist <= r + 0.5)].mean()

    return azimuthal_average


def reconstruct_full_spectrum(
    halfplane: np.ndarray, fft_shift: bool = True
) -> np.ndarray:
    """Reconstruct the full plane spectrum from a half plane spectrum.

    This is to save memory while computing. The result is the same as one would get by calling
    scipy.fft.fft2 (or similar) on the original data. The input is assumed to be the output of
    scipy.fft.rfft (or similar) and the full spectrum is expected to be square!

    By default, the spectrum is also shifted (with fft_shift)

    Parameters
    ----------
    halfplane : np.ndarray
        The half-plane array provided e.g. by scipy.fft.rfft2.
    fft_shift : bool, optional
        If True, fft-shifts the full spectrum along the last 2 axis, by default True

    Returns
    -------
    np.ndarray
        The full-plane spectrum.
    """
    dtype = halfplane.dtype
    h, w = halfplane.shape  # height, width
    full = np.zeros((h, h), dtype=dtype)  # assume h is the big dimension
    full[:, :w] = halfplane  # first half + one column

    if h % 2 == 0:
        other_half = np.roll(halfplane[::-1, ::-1][:, 1:-1].conj(), 1, axis=0)
    else:
        other_half = np.roll(halfplane[::-1, ::-1][:, :-1].conj(), 1, axis=0)

    full[:, w:] = other_half

    if fft_shift:
        return scifft.fftshift(full, axes=(-2, -1))

    return full


def image_structure_function(
    square_modulus: np.ndarray, autocorrelation: np.ndarray, lag: int
) -> np.ndarray:
    """Calculate the image structure function.

    Uses the square modulus of a (half-plane) spatial FFT timeseries and its autocorrelation
    function.

    Taken from Eq. (2) from 10.1140/epje/s10189-021-00146-2. Adjusted to use the autocorrelation
    function computed via the wiener-khinchin theorem.

    Returns the average over the whole timeseries (average over the 0th axis).

    Parameters
    ----------
    square_modulus : np.ndarray
        The square modulus of the spatial FFT of an image timeseries, e.g. np.abs(rfft2(imgs))**2.
    autocorrelation : np.ndarray
        The autocorrelation function of the spatial FFT of an image timeseries.
    lag : int
        The delay time in frame units, must be 0 <= lag <= len(square_modulus)

    Returns
    -------
    np.ndarray
        The full-plane structure function.

    Raises
    ------
    RuntimeError
        If the given lag time is longer than the timeseries.
    """
    length, *_ = square_modulus.shape

    if lag >= length:
        raise RuntimeError("Time lag cannot be longer than the timeseries itself!")

    # handling slicing with zeroth index
    if lag == 0:
        shifted_abs_square = square_modulus
        cropped_abs_square = square_modulus

    else:
        shifted_abs_square = square_modulus[lag:]
        cropped_abs_square = square_modulus[:-lag]

    autocorrelation = autocorrelation[lag].real

    sum_of_parts = (
        np.sum(shifted_abs_square + cropped_abs_square, axis=0) - 2 * autocorrelation
    )
    sum_of_parts /= length - lag  # normalization

    return reconstruct_full_spectrum(sum_of_parts)  # full plane


# convenience #####################################################################################
def normalized_rfft2(images: np.ndarray, *, workers: int = 2) -> np.ndarray:
    """Calculate the normalized rfft2.

    The normalization is the product of the last 2 dimensions of the shape of the `images` array.

    Parameters
    ----------
    images : np.ndarray
        An image sequence.
    workers : int, optional
        The number of threads to be passed to scipy.fft, by default 2

    Returns
    -------
    np.ndarray
        The normalized half-plane spatial fft of the image sequence.
    """
    *_, y, x = images.shape

    rfft2 = scifft.rfft2(images, workers=workers)
    norm = np.sqrt(x * y)
    return rfft2 / norm


def run(
    images: np.ndarray,
    lags: np.ndarray,
    keep_full_structure: bool = True,
    workers: int = 2,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Run the DDM analysis on a sequence of images.

    Parameters
    ----------
    images : np.ndarray
        The sequence of images.
    lags : np.ndarray
        An array of lag-times.
    keep_full_structure : bool, optional
        Keep and return the full image structure function, by default True
    workers : int, optional
        The number of threads to be passed to scipy.fft, by default 2

    Returns
    -------
    Tuple[np.ndarray, Optional[np.ndarray]]
        The azimuthal average of the image structure function for all lag times, and optionally the
        full image structure function for all lag times itself.
    """
    length, y, x = images.shape
    averages = np.zeros((length, y // 2))

    # setup of arrays
    if keep_full_structure:  # image structure function D(q, dt)
        dqt = np.zeros_like(images)
    else:
        dqt = None

    # spatial ffts of the images, square modulus and autocorrelation
    rfft2 = normalized_rfft2(
        images, workers=workers
    )  # scifft.rfft2(images, workers=workers)
    square_mod = np.abs(rfft2) ** 2
    autocorr = autocorrelation(rfft2, workers=workers)

    # iterate over all lags:
    for i, lag in enumerate(lags):
        sf = image_structure_function(square_mod, autocorr, lag)
        if dqt is not None:
            dqt[lag] = sf

        dist = distance_array(sf.shape)
        azimuth_avg_sf = azimuthal_average(sf, dist)
        averages[i] = azimuth_avg_sf

    return np.array(averages), dqt
