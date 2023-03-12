from typing import Union

import os
import numpy as np
from lmfit.model import Model, save_model


# create directory
model_dir = 'fit_models'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)


# ** simple exponential model
def simple_exponential(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma: float
) -> Union[np.ndarray, float]:
    """A simple exponential function model for the structure function

    .. math:
        
        D(t) = A \\left( 1 - \\exp(- \\Gamma t) \\right) + B

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

    Returns
    -------
    np.ndarray, float
        Simple exponential structure function model.
    """
    return A*(1-np.exp(-x*Gamma))+B

simple_exponential_model = Model(simple_exponential)
simple_exponential_model.set_param_hint('A', value=1.0, min=0.0, max=np.inf)
simple_exponential_model.set_param_hint('B', value=0.0, min=-np.inf, max=np.inf)
simple_exponential_model.set_param_hint('Gamma', value=1.0, min=0.0, max=np.inf)

save_model(simple_exponential_model, os.path.join(model_dir, 'simple_exponential_model.sav'))


# ** stretched exponential model
def stretched_exponential(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma: float,
    beta: float
) -> Union[np.ndarray, float]:
    """A stretched exponential function model for the structure function

    .. math:
        
        D(t) = A \\left( 1 - \\exp(- (\\Gamma t)^{\\beta}) \\right) + B

    where :math:`0 \\le \\beta \\le 1`.

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
        Stretching exponent.

    Returns
    -------
    np.ndarray, float
        Stretched exponential structure function model.
    """
    return A*(1-np.exp(-(x*Gamma)**beta))+B

stretched_exponential_model = Model(stretched_exponential)
stretched_exponential_model.set_param_hint('A', value=1.0, min=0.0, max=np.inf)
stretched_exponential_model.set_param_hint('B', value=0.0, min=-np.inf, max=np.inf)
stretched_exponential_model.set_param_hint('Gamma', value=1.0, min=0.0, max=np.inf)
stretched_exponential_model.set_param_hint('beta', value=1.0, min=0.0, max=1.0)

save_model(stretched_exponential_model, os.path.join(model_dir, 'stretched_exponential_model.sav'))


# ** compressed exponential model
def compressed_exponential(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma: float,
    beta: float
) -> Union[np.ndarray, float]:
    """A compressed exponential function model for the structure function

    .. math:
        
        D(t) = A \\left( 1 - \\exp(- (\\Gamma t)^{\\beta}) \\right) + B

    where :math:`1 \\le \\beta \\le \\infty`.

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
        Compressing exponent.

    Returns
    -------
    np.ndarray, float
        Compressed exponential structure function model.
    """
    return A*(1-np.exp(-(x*Gamma)**beta))+B

compressed_exponential_model = Model(compressed_exponential)
compressed_exponential_model.set_param_hint('A', value=1.0, min=0.0, max=np.inf)
compressed_exponential_model.set_param_hint('B', value=0.0, min=-np.inf, max=np.inf)
compressed_exponential_model.set_param_hint('Gamma', value=1.0, min=0.0, max=np.inf)
compressed_exponential_model.set_param_hint('beta', value=1.0, min=1.0, max=np.inf)

save_model(compressed_exponential_model, os.path.join(model_dir, 'compressed_exponential_model.sav'))


# ** generic exponential model
def generic_exponential(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma: float,
    beta: float
) -> Union[np.ndarray, float]:
    """A generic exponential function model for the structure function

    .. math:
        
        D(t) = A \\left( 1 - \\exp(- (\\Gamma t)^{\\beta}) \\right) + B

    where :math:`0 \\le \\beta \\le \\infty`.

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
    return A*(1-np.exp(-(x*Gamma)**beta))+B

generic_exponential_model = Model(generic_exponential)
generic_exponential_model.set_param_hint('A', value=1.0, min=0.0, max=np.inf)
generic_exponential_model.set_param_hint('B', value=0.0, min=-np.inf, max=np.inf)
generic_exponential_model.set_param_hint('Gamma', value=1.0, min=0.0, max=np.inf)
generic_exponential_model.set_param_hint('beta', value=1.0, min=0.0, max=np.inf)

save_model(generic_exponential_model, os.path.join(model_dir, 'generic_exponential_model.sav'))


# ** double exponential model
def double_exponential(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma1: float,
    beta1: float,
    Gamma2: float,
    beta2: float,
    alpha: float
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
    return A*(1 - alpha * np.exp(-(x * Gamma1)**beta1) - (1 - alpha) * np.exp(-(x * Gamma2)**beta2)) + B

double_exponential_model = Model(double_exponential)
double_exponential_model.set_param_hint('A', value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint('B', value=0.0, min=-np.inf, max=np.inf)
double_exponential_model.set_param_hint('Gamma1', value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint('beta1', value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint('Gamma2', value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint('beta2', value=1.0, min=0.0, max=np.inf)
double_exponential_model.set_param_hint('alpha', value=1.0, min=0.0, max=1.0)

save_model(double_exponential_model, os.path.join(model_dir, 'double_exponential_model.sav'))


# ** flory-schulz distribution model
def flory_schulz(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma: float,
    sigma: float
) -> Union[np.ndarray, float]:
    """A Flory-Schulz function model for the structure function

    .. math:
        
        D(t) = A \\left( 1 - (1 + \\sigma^2 \\bar{\\Gamma} t)^{-1/\\sigma^2} \\right) + B

    where :math:`0 \\le \\sigma \\le 1`. The decay rates follow a Flory-Schulz distribution

    .. math:

        G(\\Gamma) = \\frac{1}{\\bar{\\Gamma}} \\frac{(z+1)^{z+1}}{z!} \\left(\\frac{\\Gamma}{\\bar{\\Gamma}}\\right)^z \\exp\\left(-\\frac{\\Gamma}{\\bar{\\Gamma}}(z+1)\\right)

    with mean decay rate :math:`\\bar{\\Gamma}` and normalized standard deviation :math:`\\sigma=1/\\sqrt{z+1}`
    (see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

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
    return A*(1 - (1 + sigma**2 * Gamma * x)**(-1 / sigma**2)) + B

flory_schulz_model = Model(flory_schulz)
flory_schulz_model.set_param_hint('A', value=1.0, min=0.0, max=np.inf)
flory_schulz_model.set_param_hint('B', value=0.0, min=-np.inf, max=np.inf)
flory_schulz_model.set_param_hint('Gamma', value=1.0, min=0.0, max=np.inf)
flory_schulz_model.set_param_hint('sigma', value=1.0, min=0.0, max=1.0)

save_model(flory_schulz_model, os.path.join(model_dir, 'flory_schulz_model.sav'))


# ** exponential distribution model
def exponential_distribution(
    x: Union[np.ndarray, float],
    A: float,
    B: float,
    Gamma: float
) -> Union[np.ndarray, float]:
    """An exponential distribution function model for the structure function

    .. math:
        
        D(t) = A \\left( 1 - (1 + \\bar{\\Gamma} t)^{-1} \\right) + B

    The decay rates follow an exponential distribution

    .. math:

        G(\\Gamma) = \\frac{1}{\\bar{\\Gamma}} \\exp\\left(-\\frac{\\Gamma}{\\bar{\\Gamma}}\\right)

    with mean decay rate :math:`\\bar{\\Gamma}`
    (see `Mailer et al (2015) <https://iopscience.iop.org/article/10.1088/0953-8984/27/14/145102>`_).

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

    Returns
    -------
    np.ndarray, float
        Exponential distribution structure function model.
    """
    return A*(1 - 1/(1 + Gamma * x)) + B

exponential_distribution_model = Model(exponential_distribution)
exponential_distribution_model.set_param_hint('A', value=1.0, min=0.0, max=np.inf)
exponential_distribution_model.set_param_hint('B', value=0.0, min=-np.inf, max=np.inf)
exponential_distribution_model.set_param_hint('Gamma', value=1.0, min=0.0, max=np.inf)

save_model(exponential_distribution_model, os.path.join(model_dir, 'exponential_distribution_model.sav'))
