# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Mike Chen and Enrico Lattuada
# Maintainer: Enrico Lattuada

from typing import Union, Optional
import warnings
import numpy as np

from fastddm.imagestructurefunction import ImageStructureFunction
from fastddm.azimuthalaverage import AzimuthalAverage
from fastddm._config import DTYPE


def estimate_camera_noise(
        obj: Union[ImageStructureFunction, AzimuthalAverage],
        mode: str="zero",
        **kwargs
        ) -> np.ndarray:
    """
    Estimate of noise factor in ImageStructureFunction or AzimuthalAverage.
    Possible modes are:

    - "zero": zero at all wavevectors.
    - "min": minimum value at minimum lag time.
    - "high_q": takes k_min and k_max (optional, defaults are None) float parameters. It
    returns the average value of the data (calculated using all points over tau axis) in the range
    [k_min, k_max].
    - "power_spec": takes k_min and k_max (optional, defaults are None) float parameters. It
    returns the average value of the image power spectrum in the range [k_min, k_max].
    - "var": takes k_min and k_max (optional, defaults are None) float parameters. It returns
    the average value of the background corrected image power spectrum (2D Fourier transform
    variance) in the range [k_min, k_max].
    - "polyfit": takes num_points (optional, default is 5) int parameter. For each wavevector,
    it returns the constant term of a quadratic polynomial fit of the first num_points points.

    In the case of ImageStructureFunction input:
    - if k_min is None, the maximum between kx and ky is assumed
    - if k_max is None, the maximum between kx and ky is assumed

    In the case of AzimuthalAverage input:
    - if k_min is None, the maximum k is assumed
    - if k_max is None, the maximum k is assumed

    If k_min > k_max, the two values are swapped and a Warning is raised.

    In the case of ImageStructureFunction input, a boolean mask can be optionally provided for the
    following modes: "min", "high_q", "power_spec", "var". This mask is used to exclude grid points
    from the noise evaluation (where False is set). The mask array must have the same y,x shape of 
    data. If mask is not of boolean type, it is cast to bool and a warning is raised.

    Parameters
    ----------
    obj : Union[ImageStructureFunction, AzimuthalAverage]
        ImageStructureFunction or AzimuthalAverage object.
    mode : str, optional
        Estimate mode. Possible values are: "zero", "min", "high_q", "power_spec", "var", and
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
    
    if isinstance(obj, ImageStructureFunction):
        camera_noise = _estimate_noise_img_str_func(obj=obj, mode=mode, **kwargs)
    elif isinstance(obj, AzimuthalAverage):
        camera_noise = _estimate_noise_az_avg(obj=obj, mode=mode, **kwargs)
    else:
        raise TypeError(f'Input type {type(obj)} not supported.')

    return camera_noise


def _estimate_noise_az_avg(
        obj: AzimuthalAverage,
        mode: str="zero",
        **kwargs
        ) -> np.ndarray:
    """Wrapper function for the noise factor estimate for an AzimuthalAverage object.

    Parameters
    ----------
    obj : AzimuthalAverage
        AzimuthalAverage object.
    mode : str, optional
        Estimate mode. Possible values are: "zero", "min", "high_q", "power_spec", "var", and
        "polyfit". Default is "zero".

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # create functions dictionary
    func = {
        "zero": _noise_zero_az_avg,
        "min": _noise_min_az_avg,
        "high_q": _noise_high_q_az_avg,
        "power_spec": _noise_power_spec_az_avg,
        "var": _noise_var_az_avg,
        "polyfit": _noise_polyfit_az_avg,
        }
    
    return func[mode](obj, **kwargs)


def _noise_zero_az_avg(obj: AzimuthalAverage) -> np.ndarray:
    """Noise factor estimate for an AzimuthalAverage object, 'zero' mode.

    Noise is zero for all k vectors.

    Parameters
    ----------
    obj : AzimuthalAverage
        AzimuthalAverage object.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    dim = len(obj.k)
    noise = np.zeros(dim, dtype=DTYPE)

    return noise


def _noise_min_az_avg(obj: AzimuthalAverage) -> np.ndarray:
    """Noise factor estimate for an AzimuthalAverage object, 'min' mode.

    Noise is given by the minimum of the AzimuthalAverage at minimum tau.

    Parameters
    ----------
    obj : AzimuthalAverage
        AzimuthalAverage object.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # the minimum tau is always in 0 since values are sorted when the image structure function
    # is calculated
    # nanmin is used to avoid nan values
    dim = len(obj.k)
    noise_value = np.nanmin(obj.data[:, 0])
    noise = np.full(dim, fill_value=noise_value, dtype=DTYPE)

    return noise


def _noise_high_q_az_avg(
        obj: AzimuthalAverage,
        k_min: Optional[float]=None,
        k_max: Optional[float]=None
        ) -> np.ndarray:
    """Noise factor estimate for an AzimuthalAverage object, 'high_q' mode.

    Noise is given by the average value of the data (calculated using all points over tau axis) in
    the range [k_min, k_max].

    Parameters
    ----------
    obj : AzimuthalAverage
        AzimuthalAverage object.
    k_min : float, optional
        Lower bound of k range. If None, the maximum `k` is assumed. Default is None.
    k_max : float, optional
        Upper bound of k range. If None, the maximum `k` is assumed. Default is None.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # sanity checks
    if k_min is None:
        k_min = np.max(obj.k)
    if k_max is None:
        k_max = np.max(obj.k)
    if k_min > k_max:
        # if k_min > k_max, swap values
        k_min, k_max = k_max, k_min
        warnings.warn("Given k_min larger than k_max. Input values swapped.")

    # get output array dimension
    dim = len(obj.k)

    # select k range with boolean mask
    bool_mask = (obj.k >= k_min) & (obj.k <= k_max)

    # compute average value and create output array
    noise_value = np.nanmean(obj.data[bool_mask])
    noise = np.full(dim, fill_value=noise_value, dtype=DTYPE)

    return noise


def _noise_power_spec_az_avg(
        obj: AzimuthalAverage,
        k_min: Optional[float]=None,
        k_max: Optional[float]=None
        ) -> np.ndarray:
    """Noise factor estimate for an AzimuthalAverage object, 'power_spec' mode.

    Noise is given by the average value of the azimuthal average of the image power spectrum in the
    range [k_min, k_max].

    Parameters
    ----------
    obj : AzimuthalAverage
        AzimuthalAverage object.
    k_min : float, optional
        Lower bound of k range. If None, the maximum `k` is assumed. Default is None.
    k_max : float, optional
        Upper bound of k range. If None, the maximum `k` is assumed. Default is None.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # sanity checks
    if k_min is None:
        k_min = np.max(obj.k)
    if k_max is None:
        k_max = np.max(obj.k)
    if k_min > k_max:
        # if k_min > k_max, swap values
        k_min, k_max = k_max, k_min
        warnings.warn("Given k_min larger than k_max. Input values swapped.")

    # get output array dimension
    dim = len(obj.k)

    # select k range with boolean mask
    bool_mask = (obj.k >= k_min) & (obj.k <= k_max)

    # compute average value and create output array
    noise_value = 2 * np.nanmean(obj.power_spec[bool_mask])
    noise = np.full(dim, fill_value=noise_value, dtype=DTYPE)

    return noise


def _noise_var_az_avg(
        obj: AzimuthalAverage,
        k_min: Optional[float]=None,
        k_max: Optional[float]=None
        ) -> np.ndarray:
    """Noise factor estimate for an AzimuthalAverage object, 'var' mode.

    Noise is given by the average value of the azimuthal average of the background corrected image
    power spectrum in the range [k_min, k_max].

    Parameters
    ----------
    obj : AzimuthalAverage
        AzimuthalAverage object.
    k_min : float, optional
        Lower bound of k range. If None, the maximum `k` is assumed. Default is None.
    k_max : float, optional
        Upper bound of k range. If None, the maximum `k` is assumed. Default is None.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # sanity checks
    if k_min is None:
        k_min = np.max(obj.k)
    if k_max is None:
        k_max = np.max(obj.k)
    if k_min > k_max:
        # if k_min > k_max, swap values
        k_min, k_max = k_max, k_min
        warnings.warn("Given k_min larger than k_max. Input values swapped.")

    # get output array dimension
    dim = len(obj.k)

    # select k range with boolean mask
    bool_mask = (obj.k >= k_min) & (obj.k <= k_max)

    # compute average value and create output array
    noise_value = 2 * np.nanmean(obj.var[bool_mask])
    noise = np.full(dim, fill_value=noise_value, dtype=DTYPE)

    return noise


def _noise_polyfit_az_avg(
        obj: AzimuthalAverage,
        num_points: int=5,
        ) -> np.ndarray:
    """Noise factor estimate for an AzimuthalAverage object, 'polyfit' mode.

    For each k, noise is given by the 0th degree term of a quadratic polynomial fit of the first
    num_points points.

    Parameters
    ----------
    obj : AzimuthalAverage
        AzimuthalAverage object.
    num_points : int, optional
        Number of initial points (over tau axis) used in fit. Must be larger than (or equal to) 3.
        Default is 5.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # polynomial degree
    deg = 2

    # sanity check
    num_points = max(deg + 1, num_points)

    # initialize noise
    noise = _noise_zero_az_avg(obj)

    # loop through k values and fit with polynomial
    # the noise factor is estimated as the 0th degree coefficient
    dim = len(obj.k)
    for k_idx in range(dim):
        if np.isnan(obj.data[k_idx, 0]):
            noise[k_idx] = np.nan
        else:
            x = obj.tau[:num_points]
            y = obj.data[k_idx, :num_points]
            p = np.polyfit(x, y, deg=deg)
            noise[k_idx] = p[-1]
    
    return noise


def _estimate_noise_img_str_func(
        obj: ImageStructureFunction,
        mode: str="zero",
        **kwargs
        ) -> np.ndarray:
    """Wrapper function for the noise factor estimate for an ImageStructureFunction object.

    Parameters
    ----------
    obj : ImageStructureFunction
        ImageStructureFunction object.
    mode : str, optional
        Estimate mode. Possible values are: "zero", "min", "high_q", "power_spec", "var", and
        "polyfit". Default is "zero".

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # create functions dictionary
    func = {
        "zero": _noise_zero_img_str_func,
        "min": _noise_min_img_str_func,
        "high_q": _noise_high_q_img_str_func,
        "power_spec": _noise_power_spec_img_str_func,
        "var": _noise_var_img_str_func,
        "polyfit": _noise_polyfit_img_str_func,
        }
    
    return func[mode](obj, **kwargs)


def _noise_zero_img_str_func(obj: ImageStructureFunction) -> np.ndarray:
    """Noise factor estimate for an ImageStructureFunction object, 'zero' mode.

    Noise is zero for all (ky, kx) vectors.

    Parameters
    ----------
    obj : ImageStructureFunction
        ImageStructureFunction object.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    dim_t, dim_y, dim_x = obj.shape
    noise = np.zeros((dim_y, dim_x), dtype=DTYPE)

    return noise


def _noise_min_img_str_func(
        obj: ImageStructureFunction,
        mask: Optional[np.ndarray]=None
        ) -> np.ndarray:
    """Noise factor estimate for an ImageStructureFunction object, 'min' mode.

    Noise is given by the minimum of the ImageStructureFunction at minimum tau.

    Parameters
    ----------
    obj : ImageStructureFunction
        ImageStructureFunction object.
    mask : np.ndarray, optional
        If a boolean mask is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of the data. If mask is not of boolean type, it is cast to bool
        and a warning is raised. Default is None.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # get output array dimensions
    dim_t, dim_y, dim_x = obj.shape

    # check mask
    if mask is None:
        mask = np.full((dim_y, dim_x), True)
    elif mask.dtype != bool:
        mask = mask.astype(bool)
        warnings.warn("Given mask not of boolean type. Casting to bool.")

    # the minimum tau is always at index 0 on axis 0 since tau values are sorted
    # nanmin is used to avoid nan values
    noise_value = np.nanmin(obj.data[0, mask])
    noise = np.full((dim_y, dim_x), fill_value=noise_value, dtype=DTYPE)

    return noise


def _noise_high_q_img_str_func(
        obj: ImageStructureFunction,
        k_min: Optional[float]=None,
        k_max: Optional[float]=None,
        mask: Optional[np.ndarray]=None
        ) -> np.ndarray:
    """Noise factor estimate for an ImageStructureFunction object, 'high_q' mode.

    Noise is given by the average value of the data (calculated using all points over tau axis) in
    the range [k_min, k_max].

    Parameters
    ----------
    obj : ImageStructureFunction
        ImageStructureFunction object.
    k_min : float, optional
        Lower bound of k range. If None, the maximum of kx and ky is assumed. Default is None.
    k_max : float, optional
        Upper bound of k range. If None, the maximum of kx and ky is assumed. Default is None.
    mask : np.ndarray, optional
        If a boolean mask is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of the data. If mask is not of boolean type, it is cast to bool
        and a warning is raised. Default is None.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # sanity checks
    if k_min is None:
        kx_max = np.max(obj.kx)
        ky_max = np.max(obj.ky)
        k_min = max(kx_max, ky_max)
    if k_max is None:
        kx_max = np.max(obj.kx)
        ky_max = np.max(obj.ky)
        k_max = max(kx_max, ky_max)
    if k_min > k_max:
        # if k_min > k_max, swap values
        k_min, k_max = k_max, k_min
        warnings.warn("Given k_min larger than k_max. Input values swapped.")
    
    # get output array dimensions
    dim_t, dim_y, dim_x = obj.shape

    # check mask
    if mask is None:
        mask = np.full((dim_y, dim_x), True)
    elif mask.dtype != bool:
        mask = mask.astype(bool)
        warnings.warn("Given mask not of boolean type. Casting to bool.")

    # select k range with boolean mask
    KX, KY = np.meshgrid(obj.kx, obj.ky)
    k_modulus = np.sqrt(KX**2 + KY**2)
    bool_mask = (k_modulus >= k_min) & (k_modulus <= k_max) & mask

    # compute average value and create output array
    noise_value = np.nanmean(obj.data[:, bool_mask])
    noise = np.full((dim_y, dim_x), fill_value=noise_value, dtype=DTYPE)
    
    return noise


def _noise_power_spec_img_str_func(
        obj: ImageStructureFunction,
        k_min: Optional[float]=None,
        k_max: Optional[float]=None,
        mask: Optional[np.ndarray]=None
        ) -> np.ndarray:
    """Noise factor estimate for an ImageStructureFunction object, 'power_spec' mode.

    Noise is given by the average value of the image power spectrum in the range [k_min, k_max].

    Parameters
    ----------
    obj : ImageStructureFunction
        ImageStructureFunction object.
    k_min : float, optional
        Lower bound of k range. If None, the maximum of kx and ky is assumed. Default is None.
    k_max : float, optional
        Upper bound of k range. If None, the maximum of kx and ky is assumed. Default is None.
    mask : np.ndarray, optional
        If a boolean mask is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of the data. If mask is not of boolean type, it is cast to bool
        and a warning is raised. Default is None.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # sanity checks
    if k_min is None:
        kx_max = np.max(obj.kx)
        ky_max = np.max(obj.ky)
        k_min = max(kx_max, ky_max)
    if k_max is None:
        kx_max = np.max(obj.kx)
        ky_max = np.max(obj.ky)
        k_max = max(kx_max, ky_max)
    if k_min > k_max:
        # if k_min > k_max, swap values
        k_min, k_max = k_max, k_min
        warnings.warn("Given k_min larger than k_max. Input values swapped.")
    
    # get output array dimensions
    dim_t, dim_y, dim_x = obj.shape

    # check mask
    if mask is None:
        mask = np.full((dim_y, dim_x), True)
    elif mask.dtype != bool:
        mask = mask.astype(bool)
        warnings.warn("Given mask not of boolean type. Casting to bool.")

    # select k range with boolean mask
    KX, KY = np.meshgrid(obj.kx, obj.ky)
    k_modulus = np.sqrt(KX**2 + KY**2)
    bool_mask = (k_modulus >= k_min) & (k_modulus <= k_max) & mask

    # compute average value and create output array
    noise_value = 2 * np.nanmean(obj.power_spec[bool_mask])
    noise = np.full((dim_y, dim_x), fill_value=noise_value, dtype=DTYPE)
    
    return noise


def _noise_var_img_str_func(
        obj: ImageStructureFunction,
        k_min: Optional[float]=None,
        k_max: Optional[float]=None,
        mask: Optional[np.ndarray]=None
        ) -> np.ndarray:
    """Noise factor estimate for an ImageStructureFunction object, 'var' mode.

    Noise is given by the average value of the background corrected image power spectrum in the
    range [k_min, k_max].

    Parameters
    ----------
    obj : ImageStructureFunction
        ImageStructureFunction object.
    k_min : float, optional
        Lower bound of k range. If None, the maximum of kx and ky is assumed. Default is None.
    k_max : float, optional
        Upper bound of k range. If None, the maximum of kx and ky is assumed. Default is None.
    mask : np.ndarray, optional
        If a boolean mask is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of the data. If mask is not of boolean type, it is cast to bool
        and a warning is raised. Default is None.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # sanity checks
    if k_min is None:
        kx_max = np.max(obj.kx)
        ky_max = np.max(obj.ky)
        k_min = max(kx_max, ky_max)
    if k_max is None:
        kx_max = np.max(obj.kx)
        ky_max = np.max(obj.ky)
        k_max = max(kx_max, ky_max)
    if k_min > k_max:
        # if k_min > k_max, swap values
        k_min, k_max = k_max, k_min
        warnings.warn("Given k_min larger than k_max. Input values swapped.")
    
    # get output array dimensions
    dim_t, dim_y, dim_x = obj.shape

    # check mask
    if mask is None:
        mask = np.full((dim_y, dim_x), True)
    elif mask.dtype != bool:
        mask = mask.astype(bool)
        warnings.warn("Given mask not of boolean type. Casting to bool.")

    # select k range with boolean mask
    KX, KY = np.meshgrid(obj.kx, obj.ky)
    k_modulus = np.sqrt(KX**2 + KY**2)
    bool_mask = (k_modulus >= k_min) & (k_modulus <= k_max) & mask

    # compute average value and create output array
    noise_value = 2 * np.nanmean(obj.var[bool_mask])
    noise = np.full((dim_y, dim_x), fill_value=noise_value, dtype=DTYPE)
    
    return noise


def _noise_polyfit_img_str_func(
        obj: ImageStructureFunction,
        num_points: int=5
        ) -> np.ndarray:
    """Noise factor estimate for an ImageStructureFunction object, 'polyfit' mode.

    For each (ky, kx), noise is given by the 0th degree term of a quadratic polynomial fit of the
    first num_points points.

    Parameters
    ----------
    obj : ImageStructureFunction
        ImageStructureFunction object.
    num_points : int, optional
        Number of initial points (over tau axis) used in fit. Must be larger than (or equal to) 3.
        Default is 5.

    Returns
    -------
    np.ndarray
        Estimated noise factor.
    """
    # polynomial degree
    deg = 2

    # sanity check
    num_points = max(deg + 1, num_points)

    # initialize noise
    noise = _noise_zero_img_str_func(obj)

    # loop through k values and fit with polynomial
    # the noise factor is estimated as the 0th degree coefficient
    dim_y, dim_x = noise.shape
    for ky_idx in range(dim_y):
        for kx_idx in range(dim_x):
            if np.isnan(obj.data[0, ky_idx, kx_idx]):
                noise[ky_idx, kx_idx] = np.nan
            else:
                x = obj.tau[:num_points]
                y = obj.data[:num_points, ky_idx, kx_idx]
                p = np.polyfit(x, y, deg=deg)
                noise[ky_idx, kx_idx] = p[-1]
    
    return noise
