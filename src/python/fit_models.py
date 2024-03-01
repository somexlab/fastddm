# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada & Fabian Krautgasser
# Maintainer: Enrico Lattuada

r"""This module contains the fit models.

The functions listed here are used to generate the corresponding ``lmfit.Model``
for the structure function or the intermediate scattering function.
See the `lmfit documentation <https://lmfit.github.io/lmfit-py/>`_ for more
information.

The model can be imported using the model name separated by underscores.
The models for the intermediate scattering function are characterized by
the presence of ``_isf_`` in the name.
For instance, to import the ``double_exponential_model`` and the
``flory_schulz_isf_model``, do as follows:

.. code-block:: python

    # import the double exponential model for the structure function
    from fastddm.fit_models import double_exponential_model

    # import the Flory-Schulz model for the intermediate scattering function
    from fastddm.fit_models import flory_schulz_isf_model


Structure function models
~~~~~~~~~~~~~~~~~~~~~~~~~

The following ``lmfit.Model`` for the structure function are provided:

generic exponential model
*************************

A generic exponential model for the structure function:

.. math::

    D(\Delta t) = A \left\{1 - \exp\left[ -(\Gamma \Delta t)^{\beta} \right] \right\} + B .

The following parameters and settings are used:

===============================================  ======  ==============  =========================
Parameter                                        Symbol  Starting value  Limits
===============================================  ======  ==============  =========================
Amplitude (:math:`A`)                            A       1               :math:`(0, \infty)`
Noise (:math:`B`)                                B       0               :math:`(-\infty, \infty)`
Relaxation rate (:math:`\Gamma`)                 Gamma   1               :math:`(0, \infty)`
Stretching/compressing exponent (:math:`\beta`)  beta    1               :math:`(0, \infty)`
===============================================  ======  ==============  =========================

The following plot highlights the differences between curves
differing only for :math:`\beta`.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    def generic_exponential(x, A, B, Gamma, beta):
        return A*(1-np.exp(-(Gamma*x)**beta))+B

    x = np.logspace(-2, 2, num=200)

    y1 = generic_exponential(x, 1, 0, 1, 1)
    plt.semilogx(x, y1, 'b-', label=r'$\beta=1$')
    y2 = generic_exponential(x, 1, 0, 1, 0.5)
    plt.semilogx(x, y2, 'r-', label=r'$\beta=0.5$')
    y3 = generic_exponential(x, 1, 0, 1, 2)
    plt.semilogx(x, y3, 'k-', label=r'$\beta=2$')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$D(\Delta t)$')
    plt.legend()
    plt.show()

simple exponential model
************************

A simple exponential model for the structure function:

.. math::

    D(\Delta t) = A \left[1 - \exp\left( - \Gamma \Delta t \right) \right] + B .

The following parameters and settings are used:

================================  ======  ==============  =========================
Parameter                         Symbol  Starting value  Limits
================================  ======  ==============  =========================
Amplitude (:math:`A`)             A       1               :math:`(0, \infty)`
Noise (:math:`B`)                 B       0               :math:`(-\infty, \infty)`
Relaxation rate (:math:`\Gamma`)  Gamma   1               :math:`(0, \infty)`
================================  ======  ==============  =========================

It is defined from the ``generic_exponential_model`` by fixing ``beta`` to 1.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    def generic_exponential(x, A, B, Gamma, beta):
        return A*(1-np.exp(-(Gamma*x)**beta))+B

    x = np.logspace(-2, 2, num=200)

    y1 = generic_exponential(x, 1, 0, 1, 1)
    plt.semilogx(x, y1, 'b-')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$D(\Delta t)$')
    plt.show()


stretched exponential model
***************************

A stretched exponential model for the structure function:

.. math::

    D(\Delta t) = A \left\{1 - \exp\left[ -(\Gamma \Delta t)^{\beta} \right] \right\} + B .

The following parameters and settings are used:

===================================  ======  ==============  =========================
Parameter                            Symbol  Starting value  Limits
===================================  ======  ==============  =========================
Amplitude (:math:`A`)                A       1               :math:`(0, \infty)`
Noise (:math:`B`)                    B       0               :math:`(-\infty, \infty)`
Relaxation rate (:math:`\Gamma`)     Gamma   1               :math:`(0, \infty)`
Stretching exponent (:math:`\beta`)  beta    1               :math:`(0, 1]`
===================================  ======  ==============  =========================

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    def generic_exponential(x, A, B, Gamma, beta):
        return A*(1-np.exp(-(Gamma*x)**beta))+B

    x = np.logspace(-2, 2, num=200)

    y1 = generic_exponential(x, 1, 0, 1, 0.5)
    plt.semilogx(x, y1, 'b-')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$D(\Delta t)$')
    plt.show()

compressed exponential model
****************************

A compressed exponential model for the structure function:

.. math::

    D(\Delta t) = A \left\{1 - \exp\left[ -(\Gamma \Delta t)^{\beta} \right] \right\} + B .

The following parameters and settings are used:

====================================  ======  ==============  =========================
Parameter                             Symbol  Starting value  Limits
====================================  ======  ==============  =========================
Amplitude (:math:`A`)                 A       1               :math:`(0, \infty)`
Noise (:math:`B`)                     B       0               :math:`(-\infty, \infty)`
Relaxation rate (:math:`\Gamma`)      Gamma   1               :math:`(0, \infty)`
Compressing exponent (:math:`\beta`)  beta    1               :math:`[1, \infty)`
====================================  ======  ==============  =========================

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    def generic_exponential(x, A, B, Gamma, beta):
        return A*(1-np.exp(-(Gamma*x)**beta))+B

    x = np.logspace(-2, 2, num=200)

    y1 = generic_exponential(x, 1, 0, 1, 2)
    plt.semilogx(x, y1, 'b-')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$D(\Delta t)$')
    plt.show()


double exponential model
************************

A generic double exponential model for the structure function:

.. math::

    D(\Delta t) =
        A \left\{ 1 - \alpha \exp\left[-(\Gamma_1 \Delta t)^{\beta_1} \right] -
        (1 - \alpha) \exp\left[-(\Gamma_2 \Delta t)^{\beta_2} \right] \right\} + B

The following parameters and settings are used:

===================================================  ======  ==============  =========================
Parameter                                            Symbol  Starting value  Limits
===================================================  ======  ==============  =========================
Amplitude (:math:`A`)                                A       1               :math:`(0, \infty)`
Noise (:math:`B`)                                    B       0               :math:`(-\infty, \infty)`
Amplitude fraction 1 (:math:`\alpha`)                alpha   1               :math:`[0, 1]`
Relaxation rate 1 (:math:`\Gamma_1`)                 Gamma1  1               :math:`(0, \infty)`
Stretching/compressing exponent 1 (:math:`\beta_1`)  beta1   1               :math:`(0, \infty)`
Relaxation rate 2 (:math:`\Gamma_2`)                 Gamma2  1               :math:`(0, \infty)`
Stretching/compressing exponent 2 (:math:`\beta_2`)  beta2   1               :math:`(0, \infty)`
===================================================  ======  ==============  =========================

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    def double_exponential(x, A, B, alpha, Gamma1, beta1, Gamma2, beta2):
        return A*(1-alpha*np.exp(-(Gamma1*x)**beta1)-(1-alpha)*np.exp(-(Gamma2*x)**beta2))+B

    x = np.logspace(-2, 2, num=200)

    y1 = double_exponential(x, 1, 0, 0.5, 0.1, 1, 10, 1)
    plt.semilogx(x, y1, 'b-')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$D(\Delta t)$')
    plt.show()


flory schulz model
******************

A model for the structure function where the
relaxation rates follow a Flory-Schulz distribution:

.. math::

    D(\Delta t) = A \left[1 - \frac{1}{(1 + \sigma^2 \bar{\Gamma} \Delta t)^{1/\sigma^2}} \right] + B

.. math::

    G(\Gamma) = \frac{1}{\bar{\Gamma}} \frac{(z+1)^{z+1}}{z!}
    \left(\frac{\Gamma}{\bar{\Gamma}}\right)^z
    \exp\left[-\frac{\Gamma}{\bar{\Gamma}}(z+1)\right]

where :math:`\bar{\Gamma}` is the average relaxation rate and
:math:`\sigma^2=1/(z+1)` is the normalized variance
(see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

The following parameters and settings are used:

==============================================  ======  ==============  =========================
Parameter                                       Symbol  Starting value  Limits
==============================================  ======  ==============  =========================
Amplitude (:math:`A`)                           A       1               :math:`(0, \infty)`
Noise (:math:`B`)                               B       0               :math:`(-\infty, \infty)`
Relaxation rate (:math:`\bar{\Gamma}`)          Gamma   1               :math:`(0, \infty)`
Normalized standard deviation (:math:`\sigma`)  sigma   1               :math:`(0, 1]`
==============================================  ======  ==============  =========================

The following plot highlights the differences between curves
differing only for :math:`\sigma`.

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    def flory_schulz(x, A, B, Gamma, sigma):
        return A*(1-(1+sigma**2*Gamma*x)**(-1/sigma**2))+B

    def exponential(x, A, B, Gamma):
        return A*(1-np.exp(-Gamma*x))+B

    x = np.logspace(-2, 2, num=200)

    y1 = flory_schulz(x, 1, 0, 1, 1)
    plt.semilogx(x, y1, 'b-', label=r'$\sigma^2=1$')
    y2 = flory_schulz(x, 1, 0, 1, np.sqrt(0.5))
    plt.semilogx(x, y2, 'r-', label=r'$\sigma^2=0.5$')
    y3 = flory_schulz(x, 1, 0, 1, np.sqrt(0.1))
    plt.semilogx(x, y3, 'k-', label=r'$\sigma^2=0.1$')
    y4 = exponential(x, 1, 0, 1)
    plt.semilogx(x, y4, '--', color='gray', label=r'exponential')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$D(\Delta t)$')
    plt.legend()
    plt.show()


exponential distribution model
******************************

A model for the structure function where the
relaxation rates follow an exponential distribution:

.. math::

    D(\Delta t) = A \left(1 - \frac{1}{1 + \bar{\Gamma} \Delta t}\right) + B

.. math::

    G(\Gamma) = \frac{1}{\bar{\Gamma}}
    \exp\left(-\frac{\Gamma}{\bar{\Gamma}}\right)

where :math:`\bar{\Gamma}` is the average relaxation rate
(see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).
It is obtained by fixing :math:`\sigma=1` (i.e., :math:`z=0`)
in the ``flory_schulz_model``.

The following parameters and settings are used:

======================================  ======  ==============  =========================
Parameter                               Symbol  Starting value  Limits
======================================  ======  ==============  =========================
Amplitude (:math:`A`)                   A       1               :math:`(0, \infty)`
Noise (:math:`B`)                       B       0               :math:`(-\infty, \infty)`
Relaxation rate (:math:`\bar{\Gamma}`)  Gamma   1               :math:`(0, \infty)`
======================================  ======  ==============  =========================

.. plot::

    import matplotlib.pyplot as plt
    import numpy as np

    def flory_schulz(x, A, B, Gamma, sigma):
        return A*(1-(1+sigma**2*Gamma*x)**(-1/sigma**2))+B

    x = np.logspace(-2, 2, num=200)

    y1 = flory_schulz(x, 1, 0, 1, 1)
    plt.semilogx(x, y1, 'b-')
    plt.xlabel(r'$\Delta t$')
    plt.ylabel(r'$D(\Delta t)$')
    plt.show()


Intermediate scattering function models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following `lmfit.Model` for the intermediate scattering function are provided:

generic exponential isf model
*****************************

A generic exponential model for the intermediate scattering function:

.. math::

    f(\Delta t) = A \exp\left[ -(\Gamma \Delta t)^{\beta} \right] .

The following parameters and settings are used:

===============================================  ======  ==============  ===================
Parameter                                        Symbol  Starting value  Limits
===============================================  ======  ==============  ===================
Amplitude (:math:`A`)                            A       1               :math:`(0, \infty)`
Relaxation rate (:math:`\Gamma`)                 Gamma   1               :math:`(0, \infty)`
Stretching/compressing exponent (:math:`\beta`)  beta    1               :math:`(0, \infty)`
===============================================  ======  ==============  ===================

simple exponential isf model
****************************

A simple exponential model for the intermediate scattering function:

.. math::

    f(\Delta t) = A \exp\left( - \Gamma \Delta t \right) .

The following parameters and settings are used:

================================  ======  ==============  ===================
Parameter                         Symbol  Starting value  Limits
================================  ======  ==============  ===================
Amplitude (:math:`A`)             A       1               :math:`(0, \infty)`
Relaxation rate (:math:`\Gamma`)  Gamma   1               :math:`(0, \infty)`
================================  ======  ==============  ===================

It is defined from the `generic_exponential_isf_model` by fixing `beta` to 1.

stretched exponential isf model
*******************************

A stretched exponential model for the intermediate scattering function:

.. math::

    f(\Delta t) = A \exp\left[ -(\Gamma \Delta t)^{\beta} \right] .

The following parameters and settings are used:

===================================  ======  ==============  ===================
Parameter                            Symbol  Starting value  Limits
===================================  ======  ==============  ===================
Amplitude (:math:`A`)                A       1               :math:`(0, \infty)`
Relaxation rate (:math:`\Gamma`)     Gamma   1               :math:`(0, \infty)`
Stretching exponent (:math:`\beta`)  beta    1               :math:`(0, 1]`
===================================  ======  ==============  ===================

compressed exponential isf model
********************************

A compressed exponential model for the intermediate scattering function:

.. math::

    f(\Delta t) = A \exp\left[ -(\Gamma \Delta t)^{\beta} \right] .

The following parameters and settings are used:

====================================  ======  ==============  ===================
Parameter                             Symbol  Starting value  Limits
====================================  ======  ==============  ===================
Amplitude (:math:`A`)                 A       1               :math:`(0, \infty)`
Relaxation rate (:math:`\Gamma`)      Gamma   1               :math:`(0, \infty)`
Compressing exponent (:math:`\beta`)  beta    1               :math:`[1, \infty)`
====================================  ======  ==============  ===================

double exponential isf model
****************************

A generic double exponential model for the intermediate scattering function:

.. math::

    f(\Delta t) =
        A \left\{ \alpha \exp\left[-(\Gamma_1 \Delta t)^{\beta_1} \right] +
        (1 - \alpha) \exp\left[-(\Gamma_2 \Delta t)^{\beta_2} \right] \right\}

The following parameters and settings are used:

===================================================  ======  ==============  ===================
Parameter                                            Symbol  Starting value  Limits
===================================================  ======  ==============  ===================
Amplitude (:math:`A`)                                A       1               :math:`(0, \infty)`
Amplitude fraction 1 (:math:`\alpha`)                alpha   1               :math:`[0, 1]`
Relaxation rate 1 (:math:`\Gamma_1`)                 Gamma1  1               :math:`(0, \infty)`
Stretching/compressing exponent 1 (:math:`\beta_1`)  beta1   1               :math:`(0, \infty)`
Relaxation rate 2 (:math:`\Gamma_2`)                 Gamma2  1               :math:`(0, \infty)`
Stretching/compressing exponent 2 (:math:`\beta_2`)  beta2   1               :math:`(0, \infty)`
===================================================  ======  ==============  ===================

flory schulz isf model
**********************

A model for the intermediate scattering function where the
relaxation rates follow a Flory-Schulz distribution:

.. math::

    f(\Delta t) = \frac{A}{(1 + \sigma^2 \bar{\Gamma} \Delta t)^{1/\sigma^2}}

.. math::

    G(\Gamma) = \frac{1}{\bar{\Gamma}} \frac{(z+1)^{z+1}}{z!}
    \left(\frac{\Gamma}{\bar{\Gamma}}\right)^z
    \exp\left[-\frac{\Gamma}{\bar{\Gamma}}(z+1)\right]

where :math:`\bar{\Gamma}` is the average relaxation rate and
:math:`\sigma^2=1/(z+1)` is the normalized variance
(see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

The following parameters and settings are used:

==============================================  ======  ==============  ===================
Parameter                                       Symbol  Starting value  Limits
==============================================  ======  ==============  ===================
Amplitude (:math:`A`)                           A       1               :math:`(0, \infty)`
Relaxation rate (:math:`\bar{\Gamma}`)          Gamma   1               :math:`(0, \infty)`
Normalized standard deviation (:math:`\sigma`)  sigma   1               :math:`(0, 1]`
==============================================  ======  ==============  ===================

exponential distribution isf model
**********************************

A model for the intermediate scattering function where the
relaxation rates follow an exponential distribution:

.. math::

    f(\Delta t) = \frac{A}{1 + \bar{\Gamma} \Delta t}

.. math::

    G(\Gamma) = \frac{1}{\bar{\Gamma}}
    \exp\left(-\frac{\Gamma}{\bar{\Gamma}}\right)

where :math:`\bar{\Gamma}` is the average relaxation rate
(see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).
It is obtained by fixing :math:`\sigma=1` (i.e., :math:`z=0`)
in the ``flory_schulz_isf_model``.

The following parameters and settings are used:

======================================  ======  ==============  ===================
Parameter                               Symbol  Starting value  Limits
======================================  ======  ==============  ===================
Amplitude (:math:`A`)                   A       1               :math:`(0, \infty)`
Relaxation rate (:math:`\bar{\Gamma}`)  Gamma   1               :math:`(0, \infty)`
======================================  ======  ==============  ===================
"""

from typing import Union
import numpy as np
from lmfit.model import Model

EPSILON = np.finfo(float).eps

# ---------------------------------------------
# -- INTERMEDIATE SCATTERING FUNCTION MODELS --
# ---------------------------------------------


def _generic_exponential_isf(
    x: Union[np.ndarray, float], A: float, Gamma: float, beta: float
) -> Union[np.ndarray, float]:
    r"""A generic exponential function model for the
    intermediate scattering function

    .. math:

        D(t) = A \exp(- (\Gamma t)^{\beta})

    where :math:`0 < \beta < \infty`.

    Can also be used to define a `simple_exponential_isf_model`,
    where :math:`\beta = 1`, a `stretched_exponential_isf_model`,
    where :math:`0 < \beta \le 1`, or a `compressed_exponential_isf_model`,
    where :math:`1 \le \beta < \infty`.

    Parameters
    ----------
    x : numpy.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    Gamma : float
        Relaxation rate.
    beta : float
        Exponent.

    Returns
    -------
    numpy.ndarray, float
        Generic exponential intermediate scattering function model.
    """
    return A * np.exp(-((x * Gamma) ** beta))


# generic exponential model
generic_exponential_isf_model = Model(_generic_exponential_isf)
generic_exponential_isf_model.set_param_hint("A",
                                             value=1.0,
                                             min=EPSILON,
                                             max=np.inf)
generic_exponential_isf_model.set_param_hint("Gamma",
                                             value=1.0,
                                             min=EPSILON,
                                             max=np.inf)
generic_exponential_isf_model.set_param_hint("beta",
                                             value=1.0,
                                             min=EPSILON,
                                             max=np.inf)


# simple exponential model
simple_exponential_isf_model = Model(_generic_exponential_isf)
simple_exponential_isf_model.set_param_hint("A",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
simple_exponential_isf_model.set_param_hint("Gamma",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
simple_exponential_isf_model.set_param_hint("beta",
                                            value=1.0,
                                            vary=False)


# stretched exponential model
stretched_exponential_isf_model = Model(_generic_exponential_isf)
stretched_exponential_isf_model.set_param_hint("A",
                                               value=1.0,
                                               min=EPSILON,
                                               max=np.inf)
stretched_exponential_isf_model.set_param_hint("Gamma",
                                               value=1.0,
                                               min=EPSILON,
                                               max=np.inf)
stretched_exponential_isf_model.set_param_hint("beta",
                                               value=1.0,
                                               min=EPSILON,
                                               max=1.0)


# compressed exponential model
compressed_exponential_isf_model = Model(_generic_exponential_isf)
compressed_exponential_isf_model.set_param_hint("A",
                                                value=1.0,
                                                min=EPSILON,
                                                max=np.inf)
compressed_exponential_isf_model.set_param_hint("Gamma",
                                                value=1.0,
                                                min=EPSILON,
                                                max=np.inf)
compressed_exponential_isf_model.set_param_hint("beta",
                                                value=1.0,
                                                min=1.0,
                                                max=np.inf)


# double exponential model
def _double_exponential_isf(
    x: Union[np.ndarray, float],
    A: float,
    Gamma1: float,
    beta1: float,
    Gamma2: float,
    beta2: float,
    alpha: float,
) -> Union[np.ndarray, float]:
    r"""A double exponential function model for the
    intermediate scattering function

    .. math:

        D(t) = A \left( \alpha \exp(- (\Gamma_1 t)^{\beta_1}) + (1 - \alpha) \exp(- (\Gamma_2 t)^{\beta_2}) \right)

    where :math:`0 \le \beta \le \infty` and :math:`0 \le \alpha \le 1`.

    Parameters
    ----------
    x : numpy.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    Gamma1 : float
        Relaxation rate of first exponential relaxation.
    beta1 : float
        Exponent of first exponential relaxation.
    Gamma2 : float
        Relaxation rate of second exponential relaxation.
    beta2 : float
        Exponent of second exponential relaxation.
    alpha : float
        Relative amplitude of the first exponential relaxation
        (over the total amplitude).

    Returns
    -------
    numpy.ndarray, float
        Double exponential intermediate scattering function model.
    """
    return A * (
        alpha * np.exp(-((x * Gamma1) ** beta1))
        + (1 - alpha) * np.exp(-((x * Gamma2) ** beta2))
    )


double_exponential_isf_model = Model(_double_exponential_isf)
double_exponential_isf_model.set_param_hint("A",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
double_exponential_isf_model.set_param_hint("Gamma1",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
double_exponential_isf_model.set_param_hint("beta1",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
double_exponential_isf_model.set_param_hint("Gamma2",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
double_exponential_isf_model.set_param_hint("beta2",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
double_exponential_isf_model.set_param_hint("alpha",
                                            value=1.0,
                                            min=0.0,
                                            max=1.0)


# ** flory-schulz distribution model
def _flory_schulz_isf(
    x: Union[np.ndarray, float], A: float, Gamma: float, sigma: float
) -> Union[np.ndarray, float]:
    r"""A Flory-Schulz function model for the
    intermediate scattering function

    .. math:

        D(t) = A (1 + \sigma^2 \bar{\Gamma} t)^{-1/\sigma^2}

    where :math:`0 \le \sigma \le 1`.
    The decay rates follow a Flory-Schulz distribution

    .. math:

        G(\Gamma) = \frac{1}{\bar{\Gamma}} \frac{(z+1)^{z+1}}{z!} \left(\frac{\Gamma}{\bar{\Gamma}}\right)^z \exp\left(-\frac{\Gamma}{\bar{\Gamma}}(z+1)\right)

    with mean decay rate :math:`\bar{\Gamma}` and normalized standard
    deviation :math:`\sigma=1/\sqrt{z+1}`
    (see
    `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

    Can also be used to define an `exponential_distribution_isf_model`, where the
    decay rates follow an exponential distribution (for :math:`\sigma=1`, i.e., :math:`z=0`)

    .. math:

        G(\Gamma) = \frac{1}{\bar{\Gamma}} \exp\left(-\frac{\Gamma}{\bar{\Gamma}}\right)

    Parameters
    ----------
    x : numpy.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    Gamma : float
        Relaxation rate.
    sigma : float
        Width of the distribution.

    Returns
    -------
    numpy.ndarray, float
        Flory-Schulz intermediate scattering function model.
    """
    return A * (1 + sigma**2 * Gamma * x) ** (-1 / sigma**2)


flory_schulz_isf_model = Model(_flory_schulz_isf)
flory_schulz_isf_model.set_param_hint("A",
                                      value=1.0,
                                      min=EPSILON,
                                      max=np.inf)
flory_schulz_isf_model.set_param_hint("Gamma",
                                      value=1.0,
                                      min=EPSILON,
                                      max=np.inf)
flory_schulz_isf_model.set_param_hint("sigma",
                                      value=1.0,
                                      min=EPSILON,
                                      max=1.0)


# ** exponential distribution model
exponential_distribution_isf_model = Model(_flory_schulz_isf)
exponential_distribution_isf_model.set_param_hint("A",
                                                  value=1.0,
                                                  min=EPSILON,
                                                  max=np.inf)
exponential_distribution_isf_model.set_param_hint("Gamma",
                                                  value=1.0,
                                                  min=EPSILON,
                                                  max=np.inf)
exponential_distribution_isf_model.set_param_hint("sigma",
                                                  value=1.0,
                                                  vary=False)


# -------------------------------
# -- STRUCTURE FUNCTION MODELS --
# -------------------------------


# ** generic exponential model
def _generic_exponential(
    x: Union[np.ndarray, float], A: float, B: float, Gamma: float, beta: float
) -> Union[np.ndarray, float]:
    r"""A generic exponential function model for the structure function

    .. math:

        D(t) = A \left( 1 - \exp(- (\Gamma t)^{\beta}) \right) + B

    where :math:`0 < \beta < \infty`.

    Can also be used to define a `simple_exponential_model`,
    where :math:`\beta = 1`, a `stretched_exponential_model`,
    where :math:`0 < \beta \le 1`, or a `compressed_exponential_model`,
    where :math:`1 \le \beta < \infty`.

    Parameters
    ----------
    x : numpy.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    B : float
        Noise.
    Gamma : float
        Relaxation rate.
    beta : float
        Exponent.

    Returns
    -------
    numpy.ndarray, float
        Generic exponential structure function model.
    """
    return A * (1 - _generic_exponential_isf(x, 1.0, Gamma, beta)) + B


generic_exponential_model = Model(_generic_exponential)
generic_exponential_model.set_param_hint("A",
                                         value=1.0,
                                         min=EPSILON,
                                         max=np.inf)
generic_exponential_model.set_param_hint("B",
                                         value=0.0,
                                         min=-np.inf,
                                         max=np.inf)
generic_exponential_model.set_param_hint("Gamma",
                                         value=1.0,
                                         min=EPSILON,
                                         max=np.inf)
generic_exponential_model.set_param_hint("beta",
                                         value=1.0,
                                         min=EPSILON,
                                         max=np.inf)


# ** simple exponential model
simple_exponential_model = Model(_generic_exponential)
simple_exponential_model.set_param_hint("A",
                                        value=1.0,
                                        min=EPSILON,
                                        max=np.inf)
simple_exponential_model.set_param_hint("B",
                                        value=0.0,
                                        min=-np.inf,
                                        max=np.inf)
simple_exponential_model.set_param_hint("Gamma",
                                        value=1.0,
                                        min=EPSILON,
                                        max=np.inf)
simple_exponential_model.set_param_hint("beta",
                                        value=1.0,
                                        vary=False)


# ** stretched exponential model
stretched_exponential_model = Model(_generic_exponential)
stretched_exponential_model.set_param_hint("A",
                                           value=1.0,
                                           min=EPSILON,
                                           max=np.inf)
stretched_exponential_model.set_param_hint("B",
                                           value=0.0,
                                           min=-np.inf,
                                           max=np.inf)
stretched_exponential_model.set_param_hint("Gamma",
                                           value=1.0,
                                           min=EPSILON,
                                           max=np.inf)
stretched_exponential_model.set_param_hint("beta",
                                           value=1.0,
                                           min=EPSILON,
                                           max=1.0)


# ** compressed exponential model
compressed_exponential_model = Model(_generic_exponential)
compressed_exponential_model.set_param_hint("A",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
compressed_exponential_model.set_param_hint("B",
                                            value=0.0,
                                            min=-np.inf,
                                            max=np.inf)
compressed_exponential_model.set_param_hint("Gamma",
                                            value=1.0,
                                            min=EPSILON,
                                            max=np.inf)
compressed_exponential_model.set_param_hint("beta",
                                            value=1.0,
                                            min=1.0,
                                            max=np.inf)


# ** double exponential model
def _double_exponential(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma1: float,
    beta1: float,
    Gamma2: float,
    beta2: float,
    alpha: float,
) -> Union[np.ndarray, float]:
    r"""A double exponential function model for the structure function

    .. math:

        D(t) = A \left( 1 - \alpha \exp(- (\Gamma_1 t)^{\beta_1}) - (1 - \alpha) \exp(- (\Gamma_2 t)^{\beta_2}) \right) + B

    where :math:`0 \le \beta \le \infty` and :math:`0 \le \alpha \le 1`.

    Parameters
    ----------
    x : numpy.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    B : float
        Noise.
    Gamma1 : float
        Relaxation rate of first exponential relaxation.
    beta1 : float
        Exponent of first exponential relaxation.
    Gamma2 : float
        Relaxation rate of second exponential relaxation.
    beta2 : float
        Exponent of second exponential relaxation.
    alpha : float
        Relative amplitude of the first exponential relaxation
        (over the total amplitude).

    Returns
    -------
    numpy.ndarray, float
        Double exponential structure function model.
    """
    return (
        A * (1 - _double_exponential_isf(x, 1.0, Gamma1, beta1, Gamma2, beta2, alpha))
        + B
    )


double_exponential_model = Model(_double_exponential)
double_exponential_model.set_param_hint("A",
                                        value=1.0,
                                        min=EPSILON,
                                        max=np.inf)
double_exponential_model.set_param_hint("B",
                                        value=0.0,
                                        min=-np.inf,
                                        max=np.inf)
double_exponential_model.set_param_hint("Gamma1",
                                        value=1.0,
                                        min=EPSILON,
                                        max=np.inf)
double_exponential_model.set_param_hint("beta1",
                                        value=1.0,
                                        min=EPSILON,
                                        max=np.inf)
double_exponential_model.set_param_hint("Gamma2",
                                        value=1.0,
                                        min=EPSILON,
                                        max=np.inf)
double_exponential_model.set_param_hint("beta2",
                                        value=1.0,
                                        min=EPSILON,
                                        max=np.inf)
double_exponential_model.set_param_hint("alpha",
                                        value=1.0,
                                        min=0.0,
                                        max=1.0)


# ** flory-schulz distribution model
def _flory_schulz(
    x: Union[np.ndarray, float], A: float, B: float, Gamma: float, sigma: float
) -> Union[np.ndarray, float]:
    r"""A Flory-Schulz function model for the structure function

    .. math:

        D(t) = A \left( 1 - (1 + \sigma^2 \bar{\Gamma} t)^{-1/\sigma^2} \right) + B

    where :math:`0 \le \sigma \le 1`.
    The decay rates follow a Flory-Schulz distribution

    .. math:

        G(\Gamma) = \frac{1}{\bar{\Gamma}} \frac{(z+1)^{z+1}}{z!} \left(\frac{\Gamma}{\bar{\Gamma}}\right)^z \exp\left(-\frac{\Gamma}{\bar{\Gamma}}(z+1)\right)

    with mean decay rate :math:`\bar{\Gamma}` and normalized
    standard deviation :math:`\sigma=1/\sqrt{z+1}`
    (see
    `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

    Can also be used to define an `exponential_distribution_model`, where the
    decay rates follow an exponential distribution (for :math:`\sigma=1`, i.e., :math:`z=0`)

    .. math:

        G(\Gamma) = \frac{1}{\bar{\Gamma}} \exp\left(-\frac{\Gamma}{\bar{\Gamma}}\right)

    Parameters
    ----------
    x : numpy.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    B : float
        Noise.
    Gamma : float
        Relaxation rate.
    sigma : float
        Width of the distribution.

    Returns
    -------
    numpy.ndarray, float
        Flory-Schulz structure function model.
    """
    return A * (1 - _flory_schulz_isf(x, 1.0, Gamma, sigma)) + B


flory_schulz_model = Model(_flory_schulz)
flory_schulz_model.set_param_hint("A",
                                  value=1.0,
                                  min=EPSILON,
                                  max=np.inf)
flory_schulz_model.set_param_hint("B",
                                  value=0.0,
                                  min=-np.inf,
                                  max=np.inf)
flory_schulz_model.set_param_hint("Gamma",
                                  value=1.0,
                                  min=EPSILON,
                                  max=np.inf)
flory_schulz_model.set_param_hint("sigma",
                                  value=1.0,
                                  min=EPSILON,
                                  max=1.0)


# ** exponential distribution model
exponential_distribution_model = Model(_flory_schulz)
exponential_distribution_model.set_param_hint("A",
                                              value=1.0,
                                              min=EPSILON,
                                              max=np.inf)
exponential_distribution_model.set_param_hint("B",
                                              value=0.0,
                                              min=-np.inf,
                                              max=np.inf)
exponential_distribution_model.set_param_hint("Gamma",
                                              value=1.0,
                                              min=EPSILON,
                                              max=np.inf)
exponential_distribution_model.set_param_hint("sigma",
                                              value=1.0,
                                              vary=False)
