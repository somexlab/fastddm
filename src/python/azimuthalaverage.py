# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

"""Azimuthal average data class and methods."""

from typing import Tuple, Optional, Union, Iterable
from dataclasses import dataclass
import pickle
import os
import numpy as np
from scipy.interpolate import interp1d

from .imagestructurefunction import ImageStructureFunction
from ._io import _store_data


@dataclass
class AzimuthalAverage:
    """Azimuthal average container class.

    Parameters
    ----------
    _data : np.ndarray
        The packed data (azimuthal average of image structure function, power
        spectrum, and variance).
    k : np.ndarray
        The array of reference wavevector values in the bins.
    tau : np.ndarray
        The array of time delay values.
    bin_edges : np.ndarray
        The array of bin edges.
    
    Attributes
    ----------
    data : np.ndarray
        The azimuthal average of the 2D image structure function.
    power_spec: np.ndarray
        The azimuthal average of the average 2D power spectrum of the input
        images.
    var : np.ndarray
        The azimuthal average of the 2D variance (over time) of the
        Fourier transformed images.
    k : np.ndarray
        The array of reference wavevector values in the bins.
    tau : np.ndarray
        The array of time delay values.
    bin_edges : np.ndarray
        The array of bin edges.

    Methods
    -------
    save(*, fname, protocol) : None
        Save azimuthal average to binary file.
    resample(tau) : None
        Resample azimuthal average with new tau values.
    """

    _data : np.ndarray
    k : np.ndarray
    tau : np.ndarray
    bin_edges : np.ndarray

    @property
    def data(self) -> np.ndarray:
        """The azimuthal average of the 2D image structure function.

        Returns
        -------
        np.ndarray
            The azimuthal average data.
        """
        return self._data[:, :-2]

    @property
    def power_spec(self) -> np.ndarray:
        """The azimuthal average of the average 2D power spectrum of the input
        images.

        Returns
        -------
        np.ndarray
            The azimuthal average of the power spectrum.
        """
        return self._data[:, -2]

    @property
    def var(self) -> np.ndarray:
        """The azimuthal average of the 2D variance (over time) of the Fourier
        transformed input images.

        Returns
        -------
        np.ndarray
            The azimuthal average of the variance.
        """
        return self._data[:, -1]

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the azimuthal average data.

        Returns
        -------
        Tuple[int, int]
            The shape of the data.
        """
        return self.data.shape

    def save(
        self,
        fname : str = "analysis_blob",
        *,
        protocol : int = pickle.HIGHEST_PROTOCOL
        ) -> None:
        """Save AzimuthalAverage to binary file.
        The binary file is in fact a python pickle file.

        Parameters
        ----------
        fname : str, optional
            The full file name, by default "analysis_blob".
        protocol : int, optional
            pickle binary serialization protocol, by default
            pickle.HIGHEST_PROTOCOL.
        """
        # check name
        dir, name = os.path.split(fname)
        name = name if name.endswith(".aa.ddm") else f"{name}.aa.ddm"

        # save to file
        _store_data(self, fname=os.path.join(dir, name), protocol=protocol)

    def resample(
        self,
        tau : np.ndarray
        ) -> 'AzimuthalAverage':
        """Resample with new tau values and return a new AzimuthalAverage.

        Parameters
        ----------
        tau : np.ndarray
            New values of tau.

        Returns
        -------
        AzimuthalAverage
            The resampled azimuthal average.
        """
        # initialize data
        _data = np.zeros((len(self.k),len(tau) + 2))

        _tau = np.log(tau)

        # loop through k values
        for i in range(len(self.k)):
            # check for nan
            if np.isnan(self.data[i, 0]):
                _data[i, :-2] = np.full(len(tau), np.nan)
            else:
                # interpolate points in loglog scale
                f = interp1d(
                    x=np.log(self.tau),
                    y=np.log(self.data[i]),
                    kind='quadratic',
                    fill_value='extrapolate'
                    )
                _data[i, :-2] = np.exp(f(_tau))

        # append power_spec and var
        _data[:,-2] = self.power_spec
        _data[:,-1] = self.var

        return AzimuthalAverage(_data, self.k, tau, self.bin_edges)


def azimuthal_average(
    img_str_func : ImageStructureFunction,
    bins : Optional[Union[int,Iterable[float]]] = 10,
    range : Optional[Tuple[float, float]] = None,
    mask : Optional[np.ndarray] = None,
    weights : Optional[np.ndarray] = None
    ) -> AzimuthalAverage:
    """Compute the azimuthal average of the image structure function.

    Parameters
    ----------
    img_str_func : ImageStructureFunction
        The image structure function.
    bins : Union[int, Iterable[float]], optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10, by default). If `bins` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        `(k.min(), k.max())`, where `k` is the vector modulus computed from
        `kx` and `ky`. Values outside the range are ignored. The first element
        of the range must be less than or equal to the second.
    mask : np.ndarray, optional
        If a boolean `mask` is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of `data`.
    weights : np.ndarray, optional
        An array of weights, of the same y,x shape as `data`. Each
        value in `data` only contributes its associated weight towards
        the bin count (instead of 1).

    Returns
    -------
    AzimuthalAverage
        The azimuthal average.
    """
    return _azimuthal_average(
        data=img_str_func._data,
        tau=img_str_func.tau,
        kx=img_str_func.kx,
        ky=img_str_func.ky,
        bins=bins,
        range=range,
        mask=mask,
        weights=weights)

def _azimuthal_average(
    data : np.ndarray,
    tau : np.ndarray,
    kx : Optional[np.ndarray] = None,
    ky : Optional[np.ndarray] = None,
    bins : Optional[Union[int,Iterable[float]]] = 10,
    range : Optional[Tuple[float, float]] = None,
    mask : Optional[np.ndarray] = None,
    weights : Optional[np.ndarray] = None
    ) -> AzimuthalAverage:
    """Compute the azimuthal average of the image structure function.

    Parameters
    ----------
    data : np.ndarray
        The image structure function packed data. Second last and last frames
        should contain the average 2D power spectrum of the input images and
        the 2D variance (over time) of the 2D Fourier transformed images,
        respectively.
    tau : np.ndarray
        The array of time delay values.
    kx : np.ndarray, optional
        The array of spatial frequencies along axis x. If kx is None,
        the frequencies evaluated with
        `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx))`
        are used (`Nx = data.shape[2]`). Default is None.
    ky : np.ndarray, optional
        The array of spatial frequencies along axis y. If ky is None
        the frequencies evaluated with
        `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny))`
        are used (`Ny = data.shape[1]`). Default is None.
    bins : Union[int, Iterable[float]], optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10, by default). If `bins` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        `(k.min(), k.max())`, where `k` is the vector modulus computed from
        `kx` and `ky`. Values outside the range are ignored. The first element
        of the range must be less than or equal to the second.
    mask : np.ndarray, optional
        If a boolean `mask` is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of `data`.
    weights : np.ndarray, optional
        An array of weights, of the same y,x shape as `data`. Each
        value in `data` only contributes its associated weight towards
        the bin count (instead of 1).

    Returns
    -------
    AzimuthalAverage
        The azimuthal average.

    Raises
    ------
    ValueError
        If tau, kx, and ky are not compatible with shape of data.
    """

    # check input arguments
    if tau is None:
        raise ValueError("`tau` must be given for non-`ImageStructureFunction` data input.")
    elif (len(tau) + 2) != len(data):
        raise ValueError("Length of `tau` not compatible with shape of `data`.")

    # read actual image structure function shape
    dim_t, dim_y, dim_x = data.shape

    # check kx and ky
    if kx is None:
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_x))
    elif len(kx) != data.shape[2]:
        raise ValueError("Length of `kx` not compatible with shape of `data`.")
    if ky is None:
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_y))
    elif len(ky) != data.shape[1]:
        raise ValueError("Length of `ky` not compatible with shape of `data`.")

    # compute the k modulus
    X, Y = np.meshgrid(kx, ky)
    k_modulus = np.sqrt(X ** 2 + Y ** 2)

    # check range
    if range is None:
        k_min = np.min(k_modulus)
        k_max = np.max(k_modulus)
    else:
        k_min, k_max = range

    # check mask
    if mask is None:
        mask = np.full((dim_y, dim_x), True)

    # compute bin edges and initialize k
    if isinstance(bins, int):
        bin_edges = np.linspace(k_min, k_max, bins)
        k = np.zeros(bins, dtype=np.float64)
    else:   # bins is an iterable
        bin_edges = [k_min]
        for bin_width in bins:
            bin_edges.append(bin_edges[-1] + bin_width)
        k = np.zeros(len(bins), dtype=np.float64)
        bins = len(bins) + 1

    # initialize azimuthal average
    az_avg = np.full((bins, dim_t), fill_value=np.nan, dtype=np.float64)

    # loop over bins
    for i, curr_bin_edge in enumerate(bin_edges):
        if i > 0:
            e_inf = bin_edges[i-1]
            e_sup = curr_bin_edge
            curr_px = (k_modulus > e_inf) & (k_modulus <= e_sup) & mask
        else:
            curr_px = (k_modulus == curr_bin_edge) & mask

        if np.all(np.logical_not(curr_px)):
            if i > 0:
                e_inf = bin_edges[i-1]
                e_sup = curr_bin_edge
                k[i] = (e_inf + e_sup) / 2.
            else:
                k[0] = curr_bin_edge
        else:
            if weights is not None:
                num = (k_modulus[curr_px] * weights[curr_px]).mean()
                den = weights[curr_px].mean()
                k[i] = num / den
                w_avg = (data[:, curr_px] * weights[curr_px]).mean(axis=-1)
                az_avg[i] = w_avg / den
            else:
                k[i] = k_modulus[curr_px].mean()
                az_avg[i] = data[:, curr_px].mean(axis=-1)

    return AzimuthalAverage(az_avg, k, tau.astype(np.float64), bin_edges)


def melt(
    az_avg1 : AzimuthalAverage,
    az_avg2 : AzimuthalAverage
    ) -> AzimuthalAverage:
    """Melt two azimuthal averages into one object.

    Parameters
    ----------
    az_avg1 : AzimuthalAverage
        One AzimuthalAverage object.
    az_avg2 : AzimuthalAverage
        Another AzimuthalAverage object.

    Returns
    -------
    AzimuthalAverage
        The two AzimuthalAverage objects, merged into a new one.
    """

    # assign fast and slow acquisition
    if az_avg1.tau[0] < az_avg2.tau[0]:
        fast, slow = az_avg1, az_avg2
    else:
        fast, slow = az_avg2, az_avg1

    # initialize tau and data
    Nt = min(10, np.sum(slow.tau < fast.tau[-1]))
    # keep data of `fast` up to (Nt/2)-th tau of `slow`
    idx = np.argmin(np.abs(fast.tau - slow.tau[Nt // 2])) + 1
    tau = np.append(fast.tau[:idx], slow.tau[Nt // 2 + 1:])
    data = np.zeros((len(fast.k), len(tau) + 2))

    t = np.log(slow.tau[:Nt])

    # loop through k values
    for i in range(len(fast.k)):
        # check for nan
        if np.any(np.isnan([fast.data[i, 0], slow.data[i, 0]])):
            data[i] = np.full(len(tau) + 2, np.nan)
        else:
            # find multiplicative factor via least squares minimization
            # interpolate in loglog scale (smoother curve)
            f = interp1d(x=np.log(fast.tau), y=np.log(fast.data[i]), kind='cubic')
            sum_xiyi = np.sum(np.exp(f(t)) * slow.data[i, :Nt])
            sum_xi2 = np.sum(np.exp(f(t)) ** 2)
            alpha = sum_xiyi / sum_xi2

            # scale fast on slow
            data[i, :-2] = np.append(fast.data[i, :idx] * alpha, slow.data[i, Nt // 2 + 1:])
            # copy power spectrum and variance from slow
            data[i, -2:] = slow._data[i, -2:]

    return AzimuthalAverage(data, fast.k, tau, fast.bin_edges)


def mergesort(
    az_avg1 : AzimuthalAverage,
    az_avg2 : AzimuthalAverage
    ) -> AzimuthalAverage:
    """Merge the values of two azimuthal averages.
    Values will then be sorted based on tau.

    Parameters
    ----------
    az_avg1 : AzimuthalAverage
        One AzimuthalAverage object.
    az_avg2 : AzimuthalAverage
        Another AzimuthalAverage object.

    Returns
    -------
    AzimuthalAverage
        The two AzimuthalAverage objects are fused into a new one.
    """
    tau = np.append(az_avg1.tau, az_avg2.tau)
    sortidx = np.argsort(tau)

    # create new data
    dim_k, dim_tau = az_avg1.shape
    data = np.zeros_like(az_avg1, shape=(dim_k, len(tau) + 2))

    # populate data
    data[:, :-2] = np.append(az_avg1.data, az_avg2.data, axis=1)[:,sortidx]

    # copy power spectrum and variance from input with longer tau
    if az_avg1.tau[-1] > az_avg2.tau[-1]:
        data[:,-2:] = az_avg1._data[:,-2:]
    else:
        data[:,-2:] = az_avg2._data[:,-2:]

    return AzimuthalAverage(data, az_avg1.k, tau[sortidx], az_avg1.bin_edges)
