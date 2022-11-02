"""The collection of python functions to perform DDM."""

from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

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
    shape: tuple[int, ...], r_centre: Optional[int] = None
) -> np.ndarray:
    """Calculate the array of distances for a given radius centre point `r_centre`.

    The last two values of the input shape are used. If the shape is not square, the `r_centre`
    argument is ignored, and a `r_centre` equal to the integer division of each dimension by 2 is
    used.

    If the input shape is square, and no centre is supplied, then `r_centre` is the integer
    division of the square dimension by 2.

    Parameters
    ----------
    shape : tuple[int, ...]
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
    azimuthal_average = np.zeros_like(radii)

    for i, r in enumerate(radii):
        azimuthal_average[i] = image[(dist >= r - 0.5) & (dist <= r + 0.5)].mean()

    return azimuthal_average
