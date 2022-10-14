import numpy as np

def blackman(shape):
    r"""Blackman window.

    In the 1D case, the equation for the periodic three-term Blackman window
    of length `N` reads:

    .. math:: W(x) = \sum_{j=0}^2 (-1)^j a_j \cos{\left( \frac{2 \pi j x}{N} \right)}

    where :math:`a_0=0.42`, :math:`a_1=0.50`, :math:`a_2=0.08`,
    and :math:`0<=x<N`.

    See: (F.J. Harris, Proc. IEEE 66, 51 (1978))[https://ieeexplore.ieee.org/document/1455106].


    Parameters
    ----------
    shape : tuple
        Image (or sequence) shape.
        Last two values are used.

    Returns
    -------
    numpy.ndarray
        A Blackman window.
    """

    a = [0.42, 0.50, 0.08]
    x = np.array([i/float(shape[-1]) for i in range(shape[-1])])
    y = np.array([i/float(shape[-2]) for i in range(shape[-2])])
    Wx = np.zeros(shape[-1])
    Wy = np.zeros(shape[-2])

    for n in range(3):
        Wx += (-1)**n * a[n] * np.cos(2.0 * np.pi * n * x)
        Wy += (-1)**n * a[n] * np.cos(2.0 * np.pi * n * y)

    Wx, Wy = np.meshgrid(Wx,Wy)

    return Wx*Wy


def blackman_exact(shape):
    r"""Exact Blackman window.

    In the 1D case, the equation for the periodic three-term exact Blackman
    window of length `N` reads:

    .. math:: W(x) = \sum_{j=0}^2 (-1)^j a_j \cos{\left( \frac{2 \pi j x}{N} \right)}

    where :math:`a_0=0.42659071`,
    :math:`a_1=0.4965062`,
    :math:`a_2=0.07684867`,
    and :math:`0<=x<N`.

    See: (F.J. Harris, Proc. IEEE 66, 51 (1978))[https://ieeexplore.ieee.org/document/1455106].


    Parameters
    ----------
    shape : tuple
        Image (or sequence) shape.
        Last two values are used.

    Returns
    -------
    numpy.ndarray
        An exact Blackman window.
    """

    a = [7938/18608, 9240/18608, 1430/18608]
    x = np.array([i/float(shape[-1]) for i in range(shape[-1])])
    y = np.array([i/float(shape[-2]) for i in range(shape[-2])])
    Wx = np.zeros(shape[-1])
    Wy = np.zeros(shape[-2])

    for n in range(3):
        Wx += (-1)**n * a[n] * np.cos(2.0 * np.pi * n * x)
        Wy += (-1)**n * a[n] * np.cos(2.0 * np.pi * n * y)

    Wx, Wy = np.meshgrid(Wx,Wy)

    return Wx*Wy


def blackman_harris(shape):
    r"""Blackman-Harris window.

    In the 1D case, the equation for the periodic four-term Blackman-Harris
    window of length `N` reads:

    .. math:: W(x) = \sum_{j=0}^3 (-1)^j a_j \cos{\left( \frac{2 \pi j x}{N} \right)}

    where :math:`a_0=0.3635819`,
    :math:`a_1=0.4891775`,
    :math:`a_2=0.1365995`,
    :math:`a_3=0.0106411`,
    and :math:`0<=x<N`.

    See: (F. Giavazzi et al., Eur. Phys. J. E 40, 97 (2017))[https://link.springer.com/article/10.1140/epje/i2017-11587-3].


    Parameters
    ----------
    shape : tuple
        Image (or sequence) shape.
        Last two values are used.

    Returns
    -------
    numpy.ndarray
        A Blackman-Harris window.
    """

    a = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
    x = np.array([i/float(shape[-1]) for i in range(shape[-1])])
    y = np.array([i/float(shape[-2]) for i in range(shape[-2])])
    Wx = np.zeros(shape[-1])
    Wy = np.zeros(shape[-2])

    for n in range(4):
        Wx += (-1)**n * a[n] * np.cos(2.0 * np.pi * n * x)
        Wy += (-1)**n * a[n] * np.cos(2.0 * np.pi * n * y)

    Wx, Wy = np.meshgrid(Wx,Wy)

    return Wx*Wy

