# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

r"""This module contains the weight functions for azimuthal average.

The weight functions can be used to perform a sector average of the
image structure function.

For example:

.. code-block:: python

    import fastddm as fddm

    # load your images here and get shape of one image (last 2 entries in the shape property)
    img_seq = fddm.read_images(...)
    shape = img_seq.shape[-2:]

    # compute the image structure function
    dqt = fddm.ddm(img_seq, range(1, len(img_seq)))

    # create a weight 'mask' to be applied to the image structure function
    weights = fddm.weights.sector_average_weight(
        full_shape=shape,
        kx=dqt.kx,
        ky=dqt.ky,
        theta_0=90,
        delta_theta=90,
        rep=4,
        kind="uniform"
    )

    # set bin number and pass the weights to the azimuthal_average function
    nbins = max(shape) // 2
    az_avg = fddm.azimuthal_average(dqt, bins=nbins, weights=weights)


.. plot::

    import matplotlib.pyplot as plt
    from fastddm.weights import sector_average_weight

    shape = (128, 128)

    # plot setup
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(ncols=2, nrows=2)
    axs = gs.subplots(sharex=True, sharey=True)
    ((ax0, ax1), (ax2, ax3)) = axs

    # top left
    im0 = ax0.imshow(sector_average_weight(shape))
    cb = plt.colorbar(im0)
    ax0.set_axis_off()
    ax0.set_title('default parameters')

    # top right
    im1 = ax1.imshow(sector_average_weight(shape, theta_0=45, delta_theta=45, rep=4))
    cb = plt.colorbar(im1)
    ax1.set_axis_off()
    ax1.set_title(r"$\theta_0=45,\ \Delta \theta=45,\ \mathrm{rep}=4$")

    # bottom left
    im2 = ax2.imshow(sector_average_weight(shape, theta_0=90, delta_theta=140, rep=2))
    cb = plt.colorbar(im2)
    ax2.set_axis_off()
    ax2.set_title(r"$\theta_0=90,\ \Delta \theta=140,\ \mathrm{rep}=2$")

    # bottom right
    im3 = ax3.imshow(sector_average_weight(shape, theta_0=90, delta_theta=140, rep=2, kind="gauss"))
    cb = plt.colorbar(im3)
    ax3.set_axis_off()
    ax3.set_title(
        r"$\theta_0=45,\ \Delta \theta=45,$" + "\n" + "$\mathrm{rep}=4,\ \mathrm{kind}=\mathrm{gauss}$"
    )

    # displaying
    fig.tight_layout()
    plt.show()

"""


from typing import Optional, Tuple
import numpy as np


def sector_average_weight(
    full_shape: Tuple[int, int],
    kx: Optional[np.ndarray] = None,
    ky: Optional[np.ndarray] = None,
    theta_0: float = 0.0,
    delta_theta: float = 90.0,
    rep: int = 2,
    kind: Optional[str] = 'uniform'
) -> np.ndarray:
    """Evaluate weights for sector azimuthal average.
    If ``kx`` or ``ky`` are not given, the half-plane representation for the 2D
    image structure function is assumed and we use

    ``kx = 2.0 * np.pi * np.fft.fftfreq(full_shape[1])[:shape[1]]``

    ``ky = 2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(full_shape[0]))``

    with ``shape[1] = full_shape[1] // 2 + 1``.

    Parameters
    ----------
    full_shape : Tuple[int, int]
        Shape of the full 2D image structure function. This is needed in order
        to correctly account for the spare column (Nyquist frequency). The
        shape of the output will be ``(full_shape[0], full_shape[1] // 2 + 1)``,
        as for the image structure function data.
    kx : numpy.ndarray, optional
        The array of spatial frequencies along axis `x`. Default is None.
    ky : numpy.ndarray, optional
        The array of spatial frequencies along axis `y`. Default is None.
    theta_0 : float, optional
        Reference main angle (in degrees). Default is 0.
    delta_theta : float, optional
        Sector width (in degrees). If ``kind`` is "gauss", it is the
        standard deviation of the gaussian function over the angles.
        Default is 90.
    rep : int, optional
        Number of equally-spaced theta angles. Default is 2.
    kind : str, optional
        Type of weight function. Supported types are "uniform" and "gauss".
        Default is "uniform".

    Returns
    -------
    numpy.ndarray
        The weights.

    Raises
    ------
    RuntimeError
        If a value for ``kind`` other than "uniform" or "gauss" is given.
    """
    # define gaussian function
    def gauss(x, mu, sigma):
        A = 1 / (sigma * np.sqrt(2 * np.pi))
        _x = ((x - mu + np.pi) % (2 * np.pi) - np.pi) / sigma
        return A * np.exp(- 0.5 * _x ** 2)

    # available sectors
    kinds = ['uniform', 'gauss']
    if kind not in kinds:
        raise RuntimeError(
            f"Unknown kind '{kind}' selected. " +
            f"Only possible options are {kinds}."
        )

    # compute actual shape
    shape = (full_shape[0], full_shape[1] // 2 + 1)

    # compute kx and ky if not given
    if kx is None or ky is None:
        kx = 2 * np.pi * np.fft.fftfreq(full_shape[1])[:shape[1]]
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(shape[0]))

    # convert theta_0 and delta_theta
    theta_0 = (theta_0 % 360.0) * 2 * np.pi / 360
    delta_theta *= 2 * np.pi / 360

    # get grid of kx, ky
    X, Y = np.meshgrid(kx, ky)

    # compute weights
    weights = np.zeros(shape, dtype=np.float64)
    ang = np.angle(X + 1j * Y)

    for i in range(rep):
        if kind == 'gauss':
            theta_ref = theta_0 + i * 2 * np.pi / rep
            weights += gauss(ang, theta_ref, delta_theta / 2)
        else:
            theta_ref = theta_0 + i * 2 * np.pi / rep
            x = ((ang - theta_ref + np.pi) % (2 * np.pi) - np.pi) / delta_theta
            weights[x ** 2 < 0.25] += 1

    return weights


def sphere_form_factor(
    shape: Tuple[int, int],
    kx: Optional[np.ndarray] = None,
    ky: Optional[np.ndarray] = None,
    R: float = 1.0,
    contrast: float = 1.0,
    kind: Optional[str] = 'amplitude'
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
