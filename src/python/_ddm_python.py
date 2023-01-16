"""The collection of python functions to perform Differential Dynamic Microscopy."""

from typing import Optional, Tuple, Callable, Dict

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
    X, Y = np.meshgrid(kx, ky)
    k_modulus = np.sqrt(X**2 + Y**2)

    return k_modulus


def azimuthal_average(
    image: np.ndarray,
    dist: Optional[np.ndarray] = None,
    radii: Optional[np.ndarray] = None,
    binsize: Optional[float] = None,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Calculate the azimuthal average for a given image.

    If no additional parameters are presented, the spatial frequencies, radii and binsize are
    automatically calculated based on fftfreq. If other spatial frequencies/radii/binsizes are to
    be used, they can be supplied as parameters.

    Parameters
    ----------
    image : np.ndarray
        The image where the azimuthal average is to be performed.
    dist : Optional[np.ndarray], optional
        Distribution of spatial frequencies on a grid, by default None
    radii : Optional[np.ndarray], optional
        The radius values for which the azimuthal average is computed, by default None
    binsize : Optional[float], optional
        The width of the averaging ring, by default None
    mask : Optional[np.ndarray], optional
        The boolean mask used to exclude grid values from the azimuthal
        average. Use True to include, False to exclude.
        Default is None.
    weights : Optional[np.ndarray], optional
        The grid weights to be used in the azimuthal average. Default is None.

    Returns
    -------
    np.ndarray
        The average values for each given radius.
    """
    y, x = image.shape
    bigside = max(y, x)

    if binsize is None:
        binsize = 1 / bigside
    halfbin = binsize / 2

    max_len = sum(divmod(bigside, 2))  # get radius of bigger side

    # setup array of distances to center of array
    if dist is None:
        kx = scifft.fftfreq(x)
        ky = scifft.fftfreq(y)
        dist = scifft.fftshift(spatial_frequency_grid(kx, ky))

    if radii is None:
        if dist is None:
            # here we can reuse the k's:
            radii = kx if kx.size >= ky.size else ky
            radii = radii[:max_len]

        else:  # create new radii values based on the binsize
            # starting with zero
            radii = np.ones(max_len) * binsize
            radii[0] = 0
            radii = np.cumsum(radii)

    if mask is None:
        mask = np.full((y,x),True)

    if weights is None:
        weights = np.ones((y,x))

    azimuthal_average = np.zeros_like(radii, dtype=np.float64)

    for i, r in enumerate(radii):
        curr_pixels = (dist >= r - halfbin) & (dist <= r + halfbin) & mask
        azimuthal_average[i] = (image[curr_pixels]*weights[curr_pixels]).mean()/weights[curr_pixels].mean()

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


def _diff_image_structure_function(
    rfft2: np.ndarray,
    rfft2_square_mod: np.ndarray,
    lag: int,
    shape: Optional[Tuple[int, int]],
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
    shape : Optional[Tuple[int, int]]
        The output shape of the full image structure function; needs to be presented for non-square input images.

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    RuntimeError
        _description_
    """
    length, *_ = rfft2.shape

    if lag >= length:
        raise RuntimeError("Time delay cannot be longer than the timeseries itself!")

    cropped_conj = rfft2[:-lag].conj()
    shifted = rfft2[lag:]
    shifted_abs_square = rfft2_square_mod[lag:]
    cropped_abs_square = rfft2_square_mod[:-lag]

    sum_of_parts = (
        shifted_abs_square + cropped_abs_square - 2 * (cropped_conj * shifted).real
    )
    dqt = np.mean(sum_of_parts, axis=0)

    return reconstruct_full_spectrum(dqt, shape=shape)


def image_structure_function(
    square_modulus: np.ndarray,
    autocorrelation: np.ndarray,
    lag: int,
    shape: Optional[Tuple[int, ...]] = None,
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
    shape : Optional[Tuple[int, ...]]
        The output shape of the full image structure function; needs to be presented for non-square
        input images.

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

    return reconstruct_full_spectrum(sum_of_parts, shape=shape)  # full plane


def _py_image_structure_function(
    images: np.ndarray,
    lags: np.ndarray,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
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
    nx : int, optional
        The number of Fourier nodes in x direction (for normalization), by default None.
    ny : int, optional
        The number of Fourier nodes in y direction (for normalization), by default None.
    mode : str, optional
        Calculate the autocorrelation function with Wiener-Khinchin theorem ('fft') or classically ('diff'), by default "fft"
    workers : int, optional
        Number of workers to be used by scipy.fft, by default 2

    Returns
    -------
    np.ndarray
        The full image structure function for all given lag times.

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
    if nx is None or ny is None:
        _, ny, nx = images.shape
    length = len(lags)
    dqt = np.zeros((length, ny, nx))
    output_shape = (ny, nx)

    # spatial fft & square modulus
    rfft2 = normalized_rfft2(images, nx, ny, workers=workers)
    square_mod = np.abs(rfft2) ** 2

    if mode == "diff":
        # just needs argument setup
        args = [rfft2, square_mod]

    else:
        # autocorrelation for fft mode
        autocorr = autocorrelation(rfft2, workers=workers)
        args = [square_mod, autocorr]

    for i, lag in enumerate(lags):
        dqt[i] = calc_dqt(*args, lag, shape=output_shape)

    return dqt


# convenience #####################################################################################
def normalized_rfft2(
    images: np.ndarray,
    nx: Optional[int] = None,
    ny: Optional[int] = None,
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
    nx : int, optional
        The number of Fourier nodes in x direction, by default None.
    ny : int, optional
        The number of Fourier nodes in y direction, by default None.
    workers : int, optional
        The number of threads to be passed to scipy.fft, by default 2.

    Returns
    -------
    np.ndarray
        The normalized half-plane spatial fft of the image sequence.
    """
    if nx is None or ny is None:
        *_, ny, nx = images.shape

    rfft2 = scifft.rfft2(images, s=(ny, nx), workers=workers)
    norm = np.sqrt(nx * ny)
    return rfft2 / norm


def run(
    images: np.ndarray,
    lags: np.ndarray,
    keep_full_structure: bool = True,
    workers: int = 2,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run the Differential Dynamic Microscopy analysis on a sequence of images.

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
        The square modulus of the normalized rfft2 data, the azimuthal average of the image
        structure function for all given lag times, and optionally the full plane image structure
        function for all given lag times as well. The latter is None if `keep_full_structure` is
        False.
    """
    *_, y, x = images.shape
    length = len(lags)
    bigside = max(y, x)  # get bigger side
    averages = np.zeros((length, bigside // 2))

    # setup of arrays
    if keep_full_structure:  # image structure function D(q, dt)
        dqt = np.zeros((length, y, x))  # image dimensions, length of lags
    else:
        dqt = None

    # spatial ffts of the images, square modulus and autocorrelation
    rfft2 = normalized_rfft2(images, workers=workers)
    square_mod = np.abs(rfft2) ** 2
    autocorr = autocorrelation(rfft2, workers=workers)

    # setup of spatial frequencies & grid
    kx = scifft.fftfreq(x)
    ky = scifft.fftfreq(y)
    spatial_freq = scifft.fftshift(
        spatial_frequency_grid(kx, ky)
    )  # equiv of old distance array

    # iterate over all lags:
    for i, lag in enumerate(lags):
        sf = image_structure_function(square_mod, autocorr, lag, shape=images.shape)
        if dqt is not None:
            dqt[i] = sf

        averages[i] = azimuthal_average(sf, dist=spatial_freq)

    return square_mod, averages, dqt
