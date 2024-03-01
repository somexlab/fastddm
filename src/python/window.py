# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""This module contains the window functions for image preprocessing.

The window functions can be used to preprocess the input images before
calculating the image structure function.

For example:

.. code-block:: python

    import fastddm as fddm

    # load your images here
    img_seq = fddm.read_images(...)

    # preprocess the images using a Blackman-Harris window
    window = fddm.window.blackman_harris(img_seq.shape)

    # this would modify the images directly...
    # img_seq = (img_seq * window).astype(img_seq.dtype)

    # ...but you can also pass the window to the ddm function
    dqt = fddm.ddm(img_seq, range(1, len(img_seq)), window=window)


.. plot::

    import matplotlib.pyplot as plt
    from fastddm.window import blackman, blackman_harris

    fig = plt.figure()
    fig.set_figwidth(2 * fig.get_figwidth())
    gs = fig.add_gridspec(ncols=2, hspace=0)
    axs = gs.subplots(sharex=True, sharey=True)
    im0 = axs[0].imshow(blackman((128,128)))
    axs[0].set_axis_off()
    axs[0].set_title('Blackman')
    im1 = axs[1].imshow(blackman_harris((128,128)))
    axs[1].set_axis_off()
    axs[1].set_title('Blackman-Harris')
    plt.show()
"""

from typing import Tuple
import numpy as np

from ._config import DTYPE


def blackman(shape: Tuple[int, ...]) -> np.ndarray:
    r"""Blackman window.

    In the 1D case, the equation for the periodic three-term exact Blackman
    window of length `N` reads:

    .. math::

        w(x) = \sum_{j=0}^2 (-1)^j a_j \cos{\left( \frac{2 \pi j x}{N} \right)}

    where :math:`0 \le x < N` and:

    .. math::

        a_0 &= 7938 / 18608 \simeq 0.42659071
        \\
        a_1 &= 9240 / 18608 \simeq 0.4965062
        \\
        a_2 &= 1430 / 18608 \simeq 0.07684867 .

    The 2D window function then is given by :math:`W(x,y)=w(x)w(y)`.

    See: `F.J. Harris, Proc. IEEE 66, 51 (1978)
    <https://ieeexplore.ieee.org/document/1455106>`_.


    Parameters
    ----------
    shape : Tuple[int, ...]
        Image (or sequence) shape.
        Last two values are used.

    Returns
    -------
    numpy.ndarray
        A Blackman window.
    """
    *rest, ydim, xdim = shape

    a = [7938 / 18608, 9240 / 18608, 1430 / 18608]
    x = np.linspace(0, (xdim - 1) / xdim, num=xdim)
    y = np.linspace(0, (ydim - 1) / ydim, num=ydim)
    Wx = np.zeros(xdim)
    Wy = np.zeros(ydim)

    for n in range(3):
        Wx += (-1) ** n * a[n] * np.cos(2.0 * np.pi * n * x)
        Wy += (-1) ** n * a[n] * np.cos(2.0 * np.pi * n * y)

    Wx, Wy = np.meshgrid(Wx, Wy)

    return (Wx * Wy).astype(DTYPE)


def blackman_harris(shape: Tuple[int, ...]) -> np.ndarray:
    r"""Blackman-Harris window.

    In the 1D case, the equation for the periodic four-term Blackman-Harris
    window of length `N` reads:

    .. math::
    
        w(x) = \sum_{j=0}^3 (-1)^j a_j \cos{\left( \frac{2 \pi j x}{N} \right)}

    where :math:`0 \le x < N` and:

    .. math::

        a_0 &= 0.3635819
        \\
        a_1 &= 0.4891775
        \\
        a_2 &= 0.1365995
        \\
        a_3 &= 0.0106411 .

    The 2D window function then is given by :math:`W(x,y)=w(x)w(y)`.

    See: `F. Giavazzi et al., Eur. Phys. J. E 40, 97 (2017)
    <https://link.springer.com/article/10.1140/epje/i2017-11587-3>`_.


    Parameters
    ----------
    shape : Tuple[int, ...]
        Image (or sequence) shape.
        Last two values are used.

    Returns
    -------
    numpy.ndarray
        A Blackman-Harris window.
    """
    *rest, ydim, xdim = shape

    a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
    x = np.linspace(0, (xdim - 1) / xdim, num=xdim)
    y = np.linspace(0, (ydim - 1) / ydim, num=ydim)
    Wx = np.zeros(xdim)
    Wy = np.zeros(ydim)

    for n in range(4):
        Wx += (-1) ** n * a[n] * np.cos(2.0 * np.pi * n * x)
        Wy += (-1) ** n * a[n] * np.cos(2.0 * np.pi * n * y)

    Wx, Wy = np.meshgrid(Wx, Wy)

    return (Wx * Wy).astype(DTYPE)
