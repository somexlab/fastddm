"""Collection of helper functions for weights of azimuthal average"""

from typing import Optional, Tuple
import numpy as np

def sector_average_weight(
    shape : Tuple[int,int],
    kx : Optional[np.ndarray] = None,
    ky : Optional[np.ndarray] = None,
    theta_0 : Optional[float] = 0.0,
    delta_theta : Optional[float] = 90.0,
    rep : Optional[int] = 2,
    kind : Optional[str] = 'uniform'
) -> np.ndarray:
    """Evaluate weights for sector azimuthal average.

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
    theta_0 : float, optional
        Reference main angle (in degrees). Default is 0.
    delta_theta : float, optional
        Gaussian width (in degrees). Default is 90.
    rep : int, optional
        Number of equally-spaced theta angles. Default is 2.
    kind : str, optional
        Type of weight function. Supported types are 'uniform' and 'gauss'.
        Default is 'gauss'.

    Returns
    -------
    weights : np.ndarray
        The weights.

    Raises
    ------
    RuntimeError
        If a value for `kind` other than "uniform" and "gauss" is given.
    """

    def gauss(x,mu,sigma):
        A = 1 / (sigma * np.sqrt(2 * np.pi))
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    kinds = ['uniform', 'gauss']
    if kind not in kinds:
        raise RuntimeError(
            f"Unknown kind '{kind}' selected. " +
            f"Only possible options are {kinds}."
        )

    if kx is None:
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(shape[1]))

    if ky is None:
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(shape[0]))

    theta_0 = (theta_0 % 360.0) * 2 * np.pi / 360
    delta_theta *= 2 * np.pi / 360

    X, Y = np.meshgrid(kx, ky)

    weights = np.zeros(shape, dtype=np.float64)
    ang = np.angle(X + 1j * Y)

    for i in rep:
        if kind == 'gauss':
            weights += gauss(
                ang,
                theta_0 + i * 2 * np.pi / float(rep),
                delta_theta
                )
        else:
            theta_min = theta_0 + i * 2 * np.pi / float(rep) - delta_theta / 2
            theta_max = theta_0 + i * 2 * np.pi / float(rep) + delta_theta / 2
            weights[(ang <= theta_max) & (ang >= theta_min)] += 1

    return weights


def sphere_form_factor(
    shape : Tuple[int,int],
    kx : Optional[np.ndarray] = None,
    ky : Optional[np.ndarray] = None,
    R : Optional[float] = 1.0,
    contrast : Optional[float] = 1.0,
    kind : Optional[str] = 'amplitude'
) -> np.ndarray:
    """Evaluate sphere form factor.

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
    R : float, optional
        The radius of the sphere. The physical units must be compatible
        with the ones of kx/ky. Default is 1.
    contrast : float, optional
        Scattering contrast. Default is 1.
    kind : str, optional
        Type of form factor. Supported types are 'amplitude' and 'intensity'.
        Default is 'amplitude'.

    Returns
    -------
    form_fact : np.ndarray
        The sphere form factor.
    """

    def sphere_factor(k, R):
        sf = np.zeros_like(k)
        with np.errstate(divide='ignore', invalid='ignore'):
            sf = (np.sin(k * R) - k * R * np.cos(k * R)) / (k * R) ** 3
        sf[k == 0] = 1 / 3
        return sf

    kinds = ['amplitude', 'intensity']
    if kind not in kinds:
        raise RuntimeError(
            f"Unknown kind '{kind}' selected. " +
            f"Only possible options are {kinds}."
        )

    if kx is None:
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(shape[1]))

    if ky is None:
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(shape[0]))

    X, Y = np.meshgrid(kx, ky)
    k_modulus = np.sqrt(X ** 2 + Y ** 2)

    form_fact = np.zeros_like(k_modulus, dtype=np.float64)

    V = 4 * np.pi * (R ** 3) / 3

    A = contrast * V
    form_fact = A * sphere_factor(k_modulus, R)
    if kind == 'intensity':
        form_fact = 4 * np.pi * form_fact ** 2

    return form_fact
