"""Collection of helper functions for masks of azimuthal average"""

from typing import Optional, Tuple
import numpy as np

def central_cross_mask(
    shape : Tuple[int,int],
    kx : Optional[np.ndarray] = None,
    ky : Optional[np.ndarray] = None
) -> np.ndarray:
    """Evaluate mask to remove central cross from azimuthal average.

    Parameters
    ----------
    shape : (int, int)
        Shape of the new array, e.g., (128, 256).
    kx : np.ndarray, optional
        The array of spatial frequencies along axis x. If kx is None,
        the frequencies evaluated with
        `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx))`
        are used (`Nx = shape[1]`). Default is None.
    ky : np.ndarray, optional
        The array of spatial frequencies along axis y. If ky is None,
        the frequencies evaluated with
        `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny))`
        are used (`Ny = shape[0]`). Default is None.

    Returns
    -------
    mask : np.ndarray
        The mask.
    """

    if kx is None:
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(shape[1]))

    if ky is None:
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(shape[0]))

    X, Y = np.meshgrid(kx, ky)

    mask = np.full(shape, True)

    mask[(X == 0.0) | (Y == 0.0)] = False

    return mask
