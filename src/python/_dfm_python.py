"""The collection of python functions to perform DDM."""

from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import scipy.fft as scifft


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


def spatial_frequency_grid(kx: np.ndarray, ky: np.ndarray) -> np.ndarray:
    """The grid of (absolute) spatial frequency values on a grid.

    If the input `kx` and `ky` are not FFT-shifted, the output of this function can be FFT-shifted
    as well.

    Parameters
    ----------
    kx : np.ndarray
        Input from np.fft.fftfreq.
    ky : np.ndarray
        Input from np.fft.fftfreq.

    Returns
    -------
    np.ndarray
        The spatial frequency grid.
    """

    def modulus(x: float, y: float) -> float:
        return np.sqrt(x**2 + y**2)

    # get output size and initialize empty distance array
    dim_x, dim_y = kx.size, ky.size
    da = np.zeros((dim_y, dim_x))

    # calculate half of dimensions and rest
    half_x, rest_x = divmod(dim_x, 2)
    half_y, rest_y = divmod(dim_y, 2)

    # create empty array 1/4 (+rest) of the final output size
    xside = half_x + rest_x
    yside = half_y + rest_y
    da_quarter = np.zeros((yside, xside))

    # fill in the modulus values of the quarter array
    for j in range(yside):
        for i in range(xside):
            da_quarter[j, i] = modulus(kx[i], ky[j])

    # if rest is non-zero in any case, the quarter array needs to be cropped at the end
    crop_y = -rest_y if rest_y != 0 else None
    crop_x = -rest_x if rest_x != 0 else None

    # fill in the final array by cropping and flipping the quarter array
    da[:yside, :xside] = da_quarter  # top left corner
    da[yside:, :xside] = da_quarter[:crop_y, :][::-1, :]  # bottom left corner
    da[:yside, xside:] = da_quarter[:, :crop_x][:, ::-1]  # top right corner
    da[yside:, xside:] = da_quarter[:crop_y, :crop_x][::-1, ::-1]  # bottom right corner

    return da


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
    halfplane: np.ndarray,
    shape: Optional[Tuple[int, ...]] = None,
    fft_shift: bool = True,
) -> np.ndarray:
    """Reconstruct the full plane spectrum from a half plane spectrum.

    This is to save memory while computing. The result is the same as one would get by calling
    scipy.fft.fft2 (or similar) on the original data. The input is assumed to be the output of
    scipy.fft.rfft (or similar). If the full spectrum is not square, the final `shape` must be
    given, otherwise a square spectrum is assumed.

    By default, the spectrum is also shifted (with fft_shift).

    Parameters
    ----------
    halfplane : np.ndarray
        The half-plane array provided e.g. by scipy.fft.rfft2.
    shape : Tuple[int, ...]
        The shape of the full spectrum.
    fft_shift : bool, optional
        If True, fft-shifts the full spectrum along the last 2 axis, by default True

    Returns
    -------
    np.ndarray
        The full-plane spectrum.
    """

    # setup of dtype and dimensions
    dtype = halfplane.dtype
    dim_y, width = halfplane.shape  # dim_y is always directly correct

    # assume square dimensions without shape, otherwise take last element to be full x dimension
    dim_x = dim_y if shape is None else shape[-1]

    # calculate rest and set the cropping size
    rest = dim_x % 2
    crop = 1 if rest == 0 else 0

    # create full
    spectrum = np.zeros((dim_y, dim_x), dtype=dtype)

    # rfft2 half
    spectrum[:, :width] = halfplane  # first half (int-half) + 1 column

    # other half; flipped in x&y, cropped, conjugated and shifted down in y by 1 pixel
    spectrum[:, width:] = np.roll(halfplane[::-1, ::-1][:, crop:-1].conj(), 1, axis=0)

    if fft_shift:
        spectrum = scifft.fftshift(spectrum, axes=(-2, -1))

    return spectrum


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
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]
        The normalized rfft2 data, the azimuthal average of the image structure function for all
        given lag times, and optionally the full plane image structure function for all given lag
        times as well. The latter is None if `keep_full_structure` is False.
    """
    _, y, x = images.shape  # pixel dimensions, only length of lags is important
    length = len(lags)
    averages = np.zeros((length, y // 2))

    # setup of arrays
    if keep_full_structure:  # image structure function D(q, dt)
        dqt = np.zeros((length, y, x))  # image dimensions, length of lags
    else:
        dqt = None

    # spatial ffts of the images, square modulus and autocorrelation
    rfft2 = normalized_rfft2(images, workers=workers)
    square_mod = np.abs(rfft2) ** 2
    autocorr = autocorrelation(rfft2, workers=workers)

    # iterate over all lags:
    for i, lag in enumerate(lags):
        sf = image_structure_function(square_mod, autocorr, lag)
        if dqt is not None:
            dqt[i] = sf

        dist = distance_array(sf.shape)
        azimuth_avg_sf = azimuthal_average(sf, dist)
        averages[i] = azimuth_avg_sf

    return rfft2, averages, dqt
