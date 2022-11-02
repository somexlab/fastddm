"""The collection of python functions to perform DDM."""

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
