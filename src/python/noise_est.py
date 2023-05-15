# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Mike Chen and Enrico Lattuada
# Maintainer: Enrico Lattuada

from typing import Union
import numpy as np

from fastddm.imagestructurefunction import ImageStructureFunction
from fastddm.azimuthalaverage import AzimuthalAverage


def estimate_camera_noise(
        obj: Union[ImageStructureFunction, AzimuthalAverage],
        mode: str="zero",
        **kwargs
        ) -> np.ndarray:
  #take q_star and q_max also as optional input arguments
  #readability!
  #type in inner functions
  #q_min to optional q_max instead of q_star
    #np.full
    #average once and only relevant data
  #quadratic_mode_azi
    #initialize np array
    #np has polyfit already
    #using degree of polyfit as optional user define input
    f"""
    Estimate of noise factor in ImageStructureFunction or AzimuthalAverage.
    Possible modes are:

    - "zero": zero at all wavevectors
    - "min": min value at minimum lag
    - "high_q": takes q_min and q_max (optional) float parameters. Returns the average value in the
    range [q_min, q_max]
    - "power_spec": takes `q_min` and `q_max` (optional) float parameters. Returns the average value of
    the image power spectrum in the range `[q_min, q_max]`
    - "var": takes `q_min` and `q_max` (optional) float parameters. Returns the average value of
    the background corrected image power spectrum (2D Fourier transform variance) in the range
    `[q_min, q_max]`
    - "polyfit": takes `num_points` (optional) int parameter. For each wavevector, returns the
    constant term of a quadratic polynomial fit of the first `num_points`

    Parameters
    ----------
    obj : Union[ImageStructureFunction, AzimuthalAverage]
        ImageStructureFunction or AzimuthalAverage object.
    mode : str
        Estimation mode. Possible values are: "zero", "min", "high_q", "power_spec", "var", and
        "polyfit". Default is "zero".

    Returns
    -------
    np.ndarray
        Estimated noise factor.

    Raises
    ------
    TypeError
        Input type not supported.
    """
    
    if type(Dataclass) is ImageStructureFunction:
        camera_noise = calculate_camera_noise_ISF(ISF_Dataclass = Dataclass,
                                                  mode = mode)
    
    elif type(Dataclass) is AzimuthalAverage:
        camera_noise = calculate_camera_noise_AZI(AZI_Dataclass = Dataclass,
                                                  mode = mode)
    
    else:
        raise Exception('Unknown Image Structure Function Data Class')

    return camera_noise

def calculate_camera_noise_AZI(
        AZI_Dataclass,
        mode = 'default',
        q_max = None,
        q_star = None):
    """
    Mode #1
    Zero mode
    """
    if mode == 'zero' or mode == 'default':
        camera_noise = zero_mode_azi(AZI_Dataclass)
    
    """
    Mode #2
    Minimum Mode
    """
    if mode == 'min':
        camera_noise = min_mode_azi(AZI_Dataclass)
        
    """
    Mode #3
    High q Value Average Mode
    """
    if mode == 'high_q':
        camera_noise = high_q_average_mode_azi(AZI_Dataclass,
                                               q_max = q_max,
                                               q_star = q_star)
        
    """
    Mode #4
    Power specturm mode
    Uncertainty quantification in DDM
    """
    if mode == 'PS':
        camera_noise = PS_mode_azi(AZI_Dataclass,
                                   q_max = q_max,
                                   q_star = q_star)
    
    """
    Mode #5
    Variance mode
    """
    if mode == 'variance':
        camera_noise = variance_mode_azi(AZI_Dataclass,
                                         q_max = q_max,
                                         q_star = q_star)
    
    """
    Mode #6
    quadratic fit mode
    """
    if mode == 'quadratic':
        camera_noise = quadratic_mode_azi(AZI_Dataclass,
                                          N_points = 5)
        
    return camera_noise

def calculate_camera_noise_ISF(
        ISF_Dataclass,
        mode = 'default',
        q_max = None,
        q_star = None):
    """
    Mode #1
    Zero mode
    """
    if mode == 'zero' or mode == 'default':
        camera_noise = zero_mode_isf(ISF_Dataclass)
    
    """
    Mode #2
    Minimum Mode
    """
    if mode == 'min':
        camera_noise = min_mode_isf(ISF_Dataclass)
        
    """
    Mode #3
    High q Value Average Mode
    """
    if mode == 'high_q':
        camera_noise = high_q_average_mode_isf(ISF_Dataclass,
                                               q_max = q_max,
                                               q_star = q_star)
        
    """
    Mode #4
    Power specturm mode
    """
    if mode == 'PS':
        camera_noise = PS_mode_isf(ISF_Dataclass,
                                   q_max = q_max,
                                   q_star = q_star)
    
    """
    Mode #5
    Variance mode
    """
    if mode == 'variance':
        camera_noise = variance_mode_isf(ISF_Dataclass,
                                         q_max = q_max,
                                         q_star = q_star)
    
    """
    Mode #6
    quadratic fit mode
    """
    if mode == 'quadratic':
        camera_noise = quadratic_mode_isf(ISF_Dataclass,
                                          N_points = 5)
        
    return camera_noise

def zero_mode_azi(AZI_Dataclass):
    """
    Return an 1-D array of zeros the same length as obj.k

    Parameters
    ----------
    AZI_Dataclass : AzimuthalAverage
        Azimuthal average container class.

    Returns
    -------
    Numpy 1-D Array
        Camera noise, set to zero.

    """
    return np.zeros(shape = np.shape(AZI_Dataclass.k))

def min_mode_azi(AZI_Dataclass):
    """
    Minimum ISF value at the lowest time delay

    Parameters
    ----------
    AZI_Dataclass : AzimuthalAverage
        Azimuthal average container class.

    Returns
    -------
    Numpy Array
        Camera noise.

    """
    camera_noise = np.min(
        AZI_Dataclass.data.take(np.argmin(AZI_Dataclass.tau), axis = -1))
    
    return np.ones(shape = np.shape(AZI_Dataclass.k)) * camera_noise

def high_q_average_mode_azi(AZI_Dataclass, q_max, q_star = None):
    """
    Average over high frequency data to obtain camera noise

    Parameters
    ----------
    AZI_Dataclass : AzimuthalAverage
        Azimuthal average container class.
    q_max : Float
        Maximum wavevector value the camera noise is evaluated at.
    q_star : Float, optional
        Minimum wavevector value the camera noise is evaluated at. The default
        is None.

    Returns
    -------
    camera_noise : Numpy Array
        Camera noise.

    """
    
    """
    Averaging the ISF over time
    """
    avg_azi = np.average(AZI_Dataclass.data, axis = -1)
    idq_max = np.abs(AZI_Dataclass.k - q_max).argmin()
    if q_star is None:
        """
        If user did not specify q_star value the average is carried out only at
        the q value closest to q_max
        """
        camera_noise = avg_azi[idq_max]
    
    else:
        """
        Average the ISF from closest value from q_star to q_max
        """
        idq_star = np.abs(AZI_Dataclass.k - q_star).argmin()
        camera_noise = np.average(avg_azi[idq_star:idq_max])
    
    return np.ones(shape = np.shape(AZI_Dataclass.k)) * camera_noise

def PS_mode_azi(AZI_Dataclass, q_max, q_star = None):
    
    return

def variance_mode_azi(AZI_Dataclass, q_max, q_star = None):
    
    return

def quadratic_mode_azi(AZI_Dataclass, N_points = 5):
    """
    Fit the first N_points points to a quadratic curve and finding the 
    y-intercept

    Parameters
    ----------
    AZI_Dataclass : AzimuthalAverage
        Azimuthal average container class.
    N_points : Int, optional
        The number of points to use in the fitting. The default is 5.

    Returns
    -------
    Numpy 1-D Array
        Camera noise as a function of q.

    """
    from scipy.optimize import curve_fit
    
    camera_noise = []
    def quadratic_func(x, a, b, c):
        return a * (x**2) + b * x + c
    
    for isf in AZI_Dataclass.data()[:, :N_points]:
        popt, pcov = curve_fit(f = quadratic_func, 
                               xdata = AZI_Dataclass.k[:N_points], 
                               ydata = isf)
        camera_noise.append(popt[-1])
        
    return camera_noise

def zero_mode_isf(ISF_Dataclass):
    """
    Return an 1-D array of zeros the same length as obj.k

    Parameters
    ----------
    ISF_Dataclass : ImageStructureFunction
        Image Structure Function container class.

    Returns
    -------
    Numpy 1-D Array
        Camera noise, set to zero.

    """
    return np.zeros(shape = ISF_Dataclass.shape[1:])

def min_mode_isf(ISF_Dataclass):
    """
    Minimum ISF value at the lowest time delay

    Parameters
    ----------
    ISF_Dataclass : ImageStructureFunction
        Image Structure Function container class.

    Returns
    -------
    Numpy Array
        Camera noise.

    """
    camera_noise = np.min(
        ISF_Dataclass.data().take(np.argmin(ISF_Dataclass.tau), axis = -1))
    
    return np.ones(shape = ISF_Dataclass.shape[1:]) * camera_noise

def high_q_average_mode_isf(ISF_Dataclass, q_max, q_star = None):
    
    return
    

def PS_mode_isf(AZI_Dataclass, q_max, q_star = None):
    
    return

def variance_mode_isf(AZI_Dataclass, q_max, q_star = None) :
    
    return

def quadratic_mode_isf(AZI_Dataclass, N_points = 5):
        
    return 
