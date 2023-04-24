# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada & Fabian Krautgasser
# Maintainer: Enrico Lattuada

from typing import Union

import numpy as np
from lmfit.model import Model

# ---------------------------------------------
# -- INTERMEDIATE SCATTERING FUNCTION MODELS --
# ---------------------------------------------


# ** generic exponential model
def _generic_exponential_isf(
    x: Union[np.ndarray, float], A: float, Gamma: float, beta: float
) -> Union[np.ndarray, float]:
    """A generic exponential function model for the intermediate scattering function

    .. math:

        D(t) = A \\exp(- (\\Gamma t)^{\\beta})

    where :math:`0 < \\beta < \\infty`.

    Can also be used to define a `simple_exponential_isf_model`, where :math:`\\beta = 1`,
    a `stretched_exponential_isf_model`, where :math:`0 < \\beta \\le 1`, or
    a `compressed_exponential_isf_model`, where :math:`1 \\le \\beta < \\infty`.

    Parameters
    ----------
    x : np.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    Gamma : float
        Relaxation rate.
    beta : float
        Exponent.

    Returns
    -------
    np.ndarray, float
        Generic exponential intermediate scattering function model.
    """
    return A * np.exp(-((x * Gamma) ** beta))


generic_exponential_isf_model = Model(_generic_exponential_isf)
generic_exponential_isf_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
generic_exponential_isf_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
generic_exponential_isf_model.set_param_hint("beta", value=1.0, min=0.0, max=np.inf)


# ** simple exponential model
simple_exponential_isf_model = Model(_generic_exponential_isf)
simple_exponential_isf_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
simple_exponential_isf_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
simple_exponential_isf_model.set_param_hint("beta", value=1.0, vary=False)


# ** stretched exponential model
stretched_exponential_isf_model = Model(_generic_exponential_isf)
stretched_exponential_isf_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
stretched_exponential_isf_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
stretched_exponential_isf_model.set_param_hint("beta", value=1.0, min=0.0, max=1.0)


# ** compressed exponential model
compressed_exponential_isf_model = Model(_generic_exponential_isf)
compressed_exponential_isf_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
compressed_exponential_isf_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
compressed_exponential_isf_model.set_param_hint("beta", value=1.0, min=1.0, max=np.inf)


# ** double exponential model
def _double_exponential_isf(
    x: Union[np.ndarray, float],
    A: float,
    Gamma1: float,
    beta1: float,
    Gamma2: float,
    beta2: float,
    alpha: float,
) -> Union[np.ndarray, float]:
    """A double exponential function model for the intermediate scattering function

    .. math:

        D(t) = A \\left( \\alpha \\exp(- (\\Gamma_1 t)^{\\beta_1}) + (1 - \\alpha) \\exp(- (\\Gamma_2 t)^{\\beta_2}) \\right)

    where :math:`0 \\le \\beta \\le \\infty` and :math:`0 \\le \\alpha \\le 1`.

    Parameters
    ----------
    x : np.ndarray, float
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
        Relative amplitude of the first exponential relaxation (over the total amplitude).

    Returns
    -------
    np.ndarray, float
        Double exponential intermediate scattering function model.
    """
    return A * (
        alpha * np.exp(-((x * Gamma1) ** beta1))
        + (1 - alpha) * np.exp(-((x * Gamma2) ** beta2))
    )


double_exponential_isf_model = Model(_double_exponential_isf)
double_exponential_isf_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
double_exponential_isf_model.set_param_hint("Gamma1", value=1.0, min=0.0, max=np.inf)
double_exponential_isf_model.set_param_hint("beta1", value=1.0, min=0.0, max=np.inf)
double_exponential_isf_model.set_param_hint("Gamma2", value=1.0, min=0.0, max=np.inf)
double_exponential_isf_model.set_param_hint("beta2", value=1.0, min=0.0, max=np.inf)
double_exponential_isf_model.set_param_hint("alpha", value=1.0, min=0.0, max=1.0)


# ** flory-schulz distribution model
def _flory_schulz_isf(
    x: Union[np.ndarray, float], A: float, Gamma: float, sigma: float
) -> Union[np.ndarray, float]:
    """A Flory-Schulz function model for the intermediate scattering function

    .. math:

        D(t) = A (1 + \\sigma^2 \\bar{\\Gamma} t)^{-1/\\sigma^2}

    where :math:`0 \\le \\sigma \\le 1`. The decay rates follow a Flory-Schulz distribution

    .. math:

        G(\\Gamma) = \\frac{1}{\\bar{\\Gamma}} \\frac{(z+1)^{z+1}}{z!} \\left(\\frac{\\Gamma}{\\bar{\\Gamma}}\\right)^z \\exp\\left(-\\frac{\\Gamma}{\\bar{\\Gamma}}(z+1)\\right)

    with mean decay rate :math:`\\bar{\\Gamma}` and normalized standard deviation :math:`\\sigma=1/\\sqrt{z+1}`
    (see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

    Can also be used to define an `exponential_distribution_isf_model`, where the
    decay rates follow an exponential distribution (for :math:`\\sigma=1`, i.e., :math:`z=0`)

    .. math:

        G(\\Gamma) = \\frac{1}{\\bar{\\Gamma}} \\exp\\left(-\\frac{\\Gamma}{\\bar{\\Gamma}}\\right)

    Parameters
    ----------
    x : np.ndarray, float
        Independent variable.
    A : float
        Amplitude.
    Gamma : float
        Relaxation rate.
    sigma : float
        Width of the distribution.

    Returns
    -------
    np.ndarray, float
        Flory-Schulz intermediate scattering function model.
    """
    return A * (1 + sigma**2 * Gamma * x) ** (-1 / sigma**2)


flory_schulz_isf_model = Model(_flory_schulz_isf)
flory_schulz_isf_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
flory_schulz_isf_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
flory_schulz_isf_model.set_param_hint("sigma", value=1.0, min=0.0, max=1.0)


# ** exponential distribution model
exponential_distribution_isf_model = Model(_flory_schulz_isf)
exponential_distribution_isf_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
exponential_distribution_isf_model.set_param_hint(
    "Gamma", value=1.0, min=0.0, max=np.inf
)
exponential_distribution_isf_model.set_param_hint("sigma", value=1.0, vary=False)


# -------------------------------
# -- STRUCTURE FUNCTION MODELS --
# -------------------------------


# ** generic exponential model
def _generic_exponential(
    x: Union[np.ndarray, float], A: float, B: float, Gamma: float, beta: float
) -> Union[np.ndarray, float]:
    """A generic exponential function model for the structure function

    .. math:

        D(t) = A \\left( 1 - \\exp(- (\\Gamma t)^{\\beta}) \\right) + B

    where :math:`0 < \\beta < \\infty`.

    Can also be used to define a `simple_exponential_model`, where :math:`\\beta = 1`,
    a `stretched_exponential_model`, where :math:`0 < \\beta \\le 1`, or
    a `compressed_exponential_model`, where :math:`1 \\le \\beta < \\infty`.

    Parameters
    ----------
    x : np.ndarray, float
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
    np.ndarray, float
        Generic exponential structure function model.
    """
    return A * (1 - _generic_exponential_isf(x, 1.0, Gamma, beta)) + B


generic_exponential_model = Model(_generic_exponential)
generic_exponential_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
generic_exponential_model.set_param_hint("B", value=0.0, min=-np.inf, max=np.inf)
generic_exponential_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
generic_exponential_model.set_param_hint("beta", value=1.0, min=0.0, max=np.inf)


# ** simple exponential model
simple_exponential_model = Model(_generic_exponential)
simple_exponential_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
simple_exponential_model.set_param_hint("B", value=0.0, min=-np.inf, max=np.inf)
simple_exponential_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
simple_exponential_model.set_param_hint("beta", value=1.0, vary=False)


# ** stretched exponential model
stretched_exponential_model = Model(_generic_exponential)
stretched_exponential_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
stretched_exponential_model.set_param_hint("B", value=0.0, min=-np.inf, max=np.inf)
stretched_exponential_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
stretched_exponential_model.set_param_hint("beta", value=1.0, min=0.0, max=1.0)


# ** compressed exponential model
compressed_exponential_model = Model(_generic_exponential)
compressed_exponential_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
compressed_exponential_model.set_param_hint("B", value=0.0, min=-np.inf, max=np.inf)
compressed_exponential_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
compressed_exponential_model.set_param_hint("beta", value=1.0, min=1.0, max=np.inf)


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
    """A double exponential function model for the structure function

    .. math:

        D(t) = A \\left( 1 - \\alpha \\exp(- (\\Gamma_1 t)^{\\beta_1}) - (1 - \\alpha) \\exp(- (\\Gamma_2 t)^{\\beta_2}) \\right) + B

    where :math:`0 \\le \\beta \\le \\infty` and :math:`0 \\le \\alpha \\le 1`.

    Parameters
    ----------
    x : np.ndarray, float
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
        Relative amplitude of the first exponential relaxation (over the total amplitude).

    Returns
    -------
    np.ndarray, float
        Double exponential structure function model.
    """
    return (
        A * (1 - _double_exponential_isf(x, 1.0, Gamma1, beta1, Gamma2, beta2, alpha))
        + B
    )


double_exponential_model = Model(_double_exponential)
double_exponential_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint("B", value=0.0, min=-np.inf, max=np.inf)
double_exponential_model.set_param_hint("Gamma1", value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint("beta1", value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint("Gamma2", value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint("beta2", value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint("alpha", value=1.0, min=0.0, max=1.0)


# ** flory-schulz distribution model
def _flory_schulz(
    x: Union[np.ndarray, float], A: float, B: float, Gamma: float, sigma: float
) -> Union[np.ndarray, float]:
    """A Flory-Schulz function model for the structure function

    .. math:

        D(t) = A \\left( 1 - (1 + \\sigma^2 \\bar{\\Gamma} t)^{-1/\\sigma^2} \\right) + B

    where :math:`0 \\le \\sigma \\le 1`. The decay rates follow a Flory-Schulz distribution

    .. math:

        G(\\Gamma) = \\frac{1}{\\bar{\\Gamma}} \\frac{(z+1)^{z+1}}{z!} \\left(\\frac{\\Gamma}{\\bar{\\Gamma}}\\right)^z \\exp\\left(-\\frac{\\Gamma}{\\bar{\\Gamma}}(z+1)\\right)

    with mean decay rate :math:`\\bar{\\Gamma}` and normalized standard deviation :math:`\\sigma=1/\\sqrt{z+1}`
    (see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

    Can also be used to define an `exponential_distribution_model`, where the
    decay rates follow an exponential distribution (for :math:`\\sigma=1`, i.e., :math:`z=0`)

    .. math:

        G(\\Gamma) = \\frac{1}{\\bar{\\Gamma}} \\exp\\left(-\\frac{\\Gamma}{\\bar{\\Gamma}}\\right)

    Parameters
    ----------
    x : np.ndarray, float
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
    np.ndarray, float
        Flory-Schulz structure function model.
    """
    return A * (1 - _flory_schulz_isf(x, 1.0, Gamma, sigma)) + B


flory_schulz_model = Model(_flory_schulz)
flory_schulz_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
flory_schulz_model.set_param_hint("B", value=0.0, min=-np.inf, max=np.inf)
flory_schulz_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
flory_schulz_model.set_param_hint("sigma", value=1.0, min=0.0, max=1.0)


# ** exponential distribution model
exponential_distribution_model = Model(_flory_schulz)
exponential_distribution_model.set_param_hint("A", value=1.0, min=0.0, max=np.inf)
exponential_distribution_model.set_param_hint("B", value=0.0, min=-np.inf, max=np.inf)
exponential_distribution_model.set_param_hint("Gamma", value=1.0, min=0.0, max=np.inf)
exponential_distribution_model.set_param_hint("sigma", value=1.0, vary=False)
