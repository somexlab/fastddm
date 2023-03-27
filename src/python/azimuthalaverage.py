# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

"""Azimuthal average data class and methods."""

from typing import Tuple, Optional, Union, Iterable, BinaryIO
from dataclasses import dataclass
import os
from sys import byteorder
import struct
import numpy as np
from scipy.interpolate import interp1d

from .imagestructurefunction import ImageStructureFunction
from ._io_common import calculate_format_size, npdtype2format, Writer, Reader, Parser


@dataclass
class AzimuthalAverage:
    """Azimuthal average container class.

    Parameters
    ----------
    _data : np.ndarray
        The packed data (azimuthal average of image structure function, power
        spectrum, and variance).
    _err : np.ndarray
        The packed uncertainties (uncertainty of the azimuthal average, power
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
    err : np.ndarray
        The uncertainty (standard deviation) in the azimuthal average of the
        2D image structure function.
    power_spec : np.ndarray
        The azimuthal average of the average 2D power spectrum of the input
        images.
    var : np.ndarray
        The azimuthal average of the 2D variance (over time) of the
        Fourier transformed images.
    power_spec_err : np.ndarray
        The uncertainty in the azimuthal average of the average 2D power
        spectrum of the input images.
    var_err : np.ndarray
        The uncertainty in the azimuthal average of the 2D variance
        (over time) of the Fourier transformed images.
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
    _err : np.ndarray
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
    def err(self) -> np.ndarray:
        """The uncertainty (standard deviation) in the azimuthal average
        of the 2D image structure function.

        Returns
        -------
        np.ndarray
            The uncertainty.
        """
        return self._err[:, :-2]

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
    def power_spec_err(self) -> np.ndarray:
        """The uncertainty in the azimuthal average of the average
        2D power spectrum of the input images.

        Returns
        -------
        np.ndarray
            The uncertainty in the azimuthal average of the power spectrum.
        """
        return self._err[:, -2]

    @property
    def var_err(self) -> np.ndarray:
        """The uncertainty in the azimuthal average of the
        2D variance (over time) of the Fourier transformed input images.

        Returns
        -------
        np.ndarray
            The uncertainty in the azimuthal average of the variance.
        """
        return self._err[:, -1]

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
        fname : str = "analysis_blob"
        ) -> None:
        """Save AzimuthalAverage to binary file.

        Parameters
        ----------
        fname : str, optional
            The full file name, by default "analysis_blob".
        """
        # check name
        dir, name = os.path.split(fname)
        name = name if name.endswith(".aa.ddm") else f"{name}.aa.ddm"

        # save to file
        with AAWriter(file=os.path.join(dir, name)) as f:
            f.write_obj(self)

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
        _data = np.zeros((len(self.k), len(tau) + 2))
        _err = np.zeros((len(self.k), len(tau) +2))

        _tau = np.log(tau)

        # loop through k values
        for i in range(len(self.k)):
            # check for nan
            if np.isnan(self.data[i, 0]):
                _data[i, :-2] = np.full(len(tau), np.nan)
                _err[i, :-2] = np.full(len(tau), np.nan)
            else:
                # interpolate points in loglog scale
                f = interp1d(
                    x=np.log(self.tau),
                    y=np.log(self.data[i]),
                    kind='quadratic',
                    fill_value='extrapolate'
                    )
                _data[i, :-2] = np.exp(f(_tau))

                # interpolate uncertainties in loglog scale
                f = interp1d(
                    x=np.log(self.tau),
                    y=np.log(self.err[i]),
                    kind='quadratic',
                    fill_value='extrapolate'
                    )
                _err[i, :-2] = np.exp(f(_tau))

        # append power_spec and var
        _data[:, -2] = self.power_spec
        _data[:, -1] = self.var
        _err[:, -2] = self.power_spec_err
        _err[:, -1] = self.var_err

        return AzimuthalAverage(_data, _err, self.k, tau, self.bin_edges)


def azimuthal_average(
    img_str_func : ImageStructureFunction,
    bins : Optional[Union[int,Iterable[float]]] = 10,
    range : Optional[Tuple[float, float]] = None,
    mask : Optional[np.ndarray] = None,
    weights : Optional[np.ndarray] = None
    ) -> AzimuthalAverage:
    """Compute the azimuthal average of the image structure function.

    For every (not masked out) :math:`k` wavevector in the :math:`i`-th bin,
    the average is calculated as

    .. math:

        \\bar{x}_i = \\frac{\\sum_k w_k x_k}{\\sum_k w_k} ,
    
    where :math:`w_k` is the weight given to the wavevector :math:`k`.
    The uncertainty is calculated as the square root of the variance for
    weighed measures

    .. math:

        \\text{Var}(x_i) = \\left( \\frac{\\sum_k w_k x_k^2}{\\sum_k w_k} - \\bar{x}_i^2 \\right) \\frac{N_i}{N_i - 1} ,
    
    where

    .. math:

        N_i = \\frac{(\\sum_k w_k)^2}{\\sum_k w_k^2} .

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
    # read actual image structure function shape
    dim_t, dim_y, dim_x = img_str_func._data.shape

    # compute the k modulus
    X, Y = np.meshgrid(img_str_func.kx, img_str_func.ky)
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
    else:   # bins is an iterable
        bin_edges = [k_min]
        for bin_width in bins:
            bin_edges.append(bin_edges[-1] + bin_width)
        bins = len(bins) + 1
    k = np.zeros(bins, dtype=np.float64)

    # initialize k values
    k[0] = bin_edges[0]
    k[1:] = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # initialize azimuthal average, error, and helper arrays
    az_avg = np.full((bins, dim_t), fill_value=np.nan, dtype=np.float64)
    err = np.full((bins, dim_t), fill_value=np.nan, dtype=np.float64)
    wk = np.zeros(bins)
    wk2 = np.zeros(bins)
    tmp = np.zeros(dim_t, dtype=np.float64)

    # loop over k vectors on half-plane
    for (i, j), k_val in np.ndenumerate(k_modulus[:, :(dim_x // 2 + 1)]):
        # count 1 if mid column (kx == 0) or
        # when dim_x is odd and we are considering the first column
        if (j == dim_x // 2) or (dim_x % 2 == 1 and j == 0):
            fact = 1.0
        else:
            fact = 2.0
        # check mask
        if mask[i, j]:
            # get current bin
            idx = np.searchsorted(bin_edges, k_val)
            if idx < bins:
                if np.isnan(az_avg[idx, 0]):
                    az_avg[idx] = np.zeros(dim_t)
                    err[idx] = np.zeros(dim_t)
                    k[idx] = 0.0
                tmp = fact * img_str_func._data[:, i, j]
                if weights is None:
                    az_avg[idx] += tmp
                    err[idx] += tmp * img_str_func._data[:, i, j]
                    k[idx] += fact * k_val
                    wk[idx] += fact
                    wk2[idx] += fact
                else:
                    tmp *= weights[i, j]
                    az_avg[idx] += tmp
                    err[idx] += tmp * img_str_func._data[:, i, j]
                    k[idx] += fact * k_val * weights[i, j]
                    wk[idx] += fact * weights[i, j]
                    wk2[idx] += fact * weights[i, j]**2

    for i, d in enumerate(wk):
        if d > 0:
            az_avg[i] /= d
            err[i] = np.sqrt((err[i] / d - az_avg[i]**2) / (1 - wk2[i] / d**2))
            k[i] /= d

    return AzimuthalAverage(az_avg, err, k, img_str_func.tau, bin_edges)


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

    For every (not masked out) :math:`k` wavevector in the :math:`i`-th bin,
    the average is calculated as

    .. math:

        \\bar{x}_i = \\frac{\\sum_k w_k x_k}{\\sum_k w_k} ,
    
    where :math:`w_k` is the weight given to the wavevector :math:`k`.
    The uncertainty is calculated as the square root of the variance for
    weighed measures

    .. math:

        \\text{Var}(x_i) = \\left( \\frac{\\sum_k w_k x_k^2}{\\sum_k w_k} - \\bar{x}_i^2 \\right) \\frac{N_i}{N_i - 1} ,
    
    where

    .. math:

        N_i = \\frac{(\\sum_k w_k)^2}{\\sum_k w_k^2} .

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
    else:   # bins is an iterable
        bin_edges = [k_min]
        for bin_width in bins:
            bin_edges.append(bin_edges[-1] + bin_width)
        bins = len(bins) + 1
    k = np.zeros(bins, dtype=np.float64)

    # initialize k values
    k[0] = bin_edges[0]
    k[1:] = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # initialize azimuthal average
    az_avg = np.full((bins, dim_t), fill_value=np.nan, dtype=np.float64)
    err = np.full((bins, dim_t), fill_value=np.nan, dtype=np.float64)
    wk = np.zeros(bins)
    wk2 = np.zeros(bins)
    tmp = np.zeros(dim_t, dtype=np.float64)

    # loop over k vectors on half-plane
    for (i, j), k_val in np.ndenumerate(k_modulus):
        # check mask
        if mask[i, j]:
            # get current bin
            idx = np.searchsorted(bin_edges, k_val)
            if idx < bins:
                if np.isnan(az_avg[idx, 0]):
                    az_avg[idx] = np.zeros(dim_t)
                    err[idx] = np.zeros(dim_t)
                    k[idx] = 0.0
                tmp = data[:, i, j]
                if weights is None:
                    az_avg[idx] += tmp
                    err[idx] += tmp * data[:, i, j]
                    k[idx] += k_val
                    wk[idx] += 1
                    wk2[idx] += 1
                else:
                    tmp *= weights[i, j]
                    az_avg[idx] += tmp
                    err[idx] += tmp * data[:, i, j]
                    k[idx] += k_val * weights[i, j]
                    wk[idx] += weights[i, j]
                    wk2[idx] += weights[i, j]**2

    for i, d in enumerate(wk):
        if d > 0:
            az_avg[i] /= d
            err[i] = np.sqrt((err[i] / d - az_avg[i]**2) / (1 - wk2[i] / d**2))
            k[i] /= d

    return AzimuthalAverage(az_avg, err, k, tau.astype(np.float64), bin_edges)


class AAWriter(Writer):
    """FastDDM azimuthal average writer class.
    Inherits from `Writer`. It adds the following unique methods:

    Methods
    -------
    write_obj(obj) : None
        Write AzimuthalAverage object to binary file.
    """
    def write_obj(
        self,
        obj : AzimuthalAverage
        ) -> None:
        """Write AzimuthalAverage object to binary file.

        Parameters
        ----------
        obj : AzimuthalAverage
            AzimuthalAverage object.
        """
        # get data dtype
        dtype = npdtype2format(obj.data.dtype.name)
        # get data shape
        Nk, Nt = obj.shape
        # assign Nextra = 2
        # we have power_spectrum + variance
        Nextra = obj._data.shape[-1] - Nt

        # write file header
        self._write_header(Nk, Nt, Nextra, dtype)

        # write data
        self._write_data(obj)

    def _write_header(
        self,
        Nk : int,
        Nt : int,
        Nextra : int,
        dtype : str
        ) -> None:
        """Write image structure function file header.

        In version 0.1, the header is structured as follows:
        * bytes 0-1: endianness (`LL` = 'little'; `BB` = 'big'), 'utf-8' encoding
        * bytes 2-3: file identifier (22), `H` (unsigned short)
        * bytes 4-5: file version as (major_version, minor_version), `BB` (unsigned char)
        * byte 6: dtype (`d` = float64; `f` = float32), 'utf-8' encoding
        * bytes 7-14: data height, `Q` (unsigned long long)
        * bytes 15-22: data width, `Q` (unsigned long long)
        * bytes 23-30: extra slices, `Q` (unsigned long long)

        Parameters
        ----------
        Nk : int
            Height.
        Nt : int
            Width.
        Nextra : int
            Number of extra slices.
        dtype : str
            Data dtype.
        """
        curr_byte_len = 0
        # system endianness
        if byteorder == 'little':
            self._fh.write('LL'.encode('utf-8'))
        else:
            self._fh.write('BB'.encode('utf-8'))
        curr_byte_len += 2

        # file identifier
        self._fh.write(struct.pack('H', 22))
        curr_byte_len += calculate_format_size('H')

        # file version
        self._fh.write(struct.pack('BB', *(self.version)))
        curr_byte_len += 2 * calculate_format_size('B')

        # dtype
        self._fh.write(dtype.encode('utf-8'))
        curr_byte_len += 1

        # height
        self._fh.write(struct.pack('Q', Nk))
        curr_byte_len += calculate_format_size('Q')

        # width
        self._fh.write(struct.pack('Q', Nt))
        curr_byte_len += calculate_format_size('Q')

        # extra slices
        self._fh.write(struct.pack('Q', Nextra))
        curr_byte_len += calculate_format_size('Q')

        # add empty bytes up to HEAD_BYTE_LEN for future use (if needed)
        self._fh.write(bytearray(self.head_byte_len - curr_byte_len))

    def _write_data(
        self,
        obj : AzimuthalAverage
        ) -> None:
        """Write azimuthal average data.

        In version 0.1, the data is stored in 'C' order and `dtype` format as follows:
        * from `data_offset`: _data
        * from `err_offset`: _err
        * from `k_offset`: `k` array
        * from `tau_offset`: `tau` array
        * from `bin_edges_offset`: `bin_edges` array

        From the end of the file,
        the byte offsets are stored in `Q` (unsigned long long) format in this order:
        * `data_offset`
        * `err_offset`
        * `k_offset`
        * `tau_offset`
        * `bin_edges_offset`

        Parameters
        ----------
        obj : AzimuthalAverage
            The azimuthal average object.
        """
        # get data format
        fmt = npdtype2format(obj.data.dtype.name)

        # get data shape
        Nk, Nt = obj._data.shape

        # write _data
        obj._data.tofile(self._fh)
        data_offset = self.head_byte_len

        # write _err
        obj._err.tofile(self._fh)
        err_offset = data_offset + Nk * Nt * calculate_format_size(fmt)

        # write kx, ky, and tau
        obj.k.tofile(self._fh)
        obj.tau.tofile(self._fh)
        obj.bin_edges.tofile(self._fh)
        k_offset = err_offset + Nk * Nt * calculate_format_size(fmt)
        tau_offset = k_offset + Nk * calculate_format_size(fmt)
        bin_edges_offset = tau_offset + Nt * calculate_format_size(fmt)

        # write byte offsets
        self._fh.write(struct.pack('Q', bin_edges_offset))
        self._fh.write(struct.pack('Q', tau_offset))
        self._fh.write(struct.pack('Q', k_offset))
        self._fh.write(struct.pack('Q', err_offset))
        self._fh.write(struct.pack('Q', data_offset))


class AAReader(Reader):
    """FastDDM azimuthal average reader class.
    Inherits from `Reader`. It adds the following unique parameters and methods:

    Methods
    -------
    load(obj) : AzimuthalAverage
        Load the azimuthal average.
    get_k() : np.ndarray
        Read k array.
    get_tau() : np.ndarray
        Read tau array.
    get_bin_edges() : np.ndarray
        Read bin_edges array.
    get_k_slice(k_index) : np.ndarray
        Read k slice from data.
    get_k_slice_err(k_index) : np.ndarray
        Read k slice uncertainty from data.
    """
    def __init__(self, file : str):
        super().__init__(file)
        self._parser = AAParser(self._fh)
        self._metadata = self._parser.read_metadata()

    def load(self) -> AzimuthalAverage:
        """Load azimuthal average from file.

        Returns
        -------
        AzimuthalAverage
            The AzimuthalAverage object.
        """
        Nk = self._metadata['Nk']
        Nt = self._metadata['Nt']
        Nextra = self._metadata['Nextra']

        return AzimuthalAverage(
            self._parser.read_array(self._metadata['data_offset'], (Nk, Nt + Nextra)),
            self._parser.read_array(self._metadata['err_offset'], (Nk, Nt + Nextra)),
            self.get_k(),
            self.get_tau(),
            self.get_bin_edges()
            )

    def get_k(self) -> np.ndarray:
        """Read k array from file.

        Returns
        -------
        np.ndarray
            The k array.
        """
        offset = self._metadata['k_offset']
        Nk = self._metadata['Nk']

        return self._parser.read_array(offset, Nk)

    def get_tau(self) -> np.ndarray:
        """Read tau array from file.

        Returns
        -------
        np.ndarray
            The tau array.
        """
        offset = self._metadata['tau_offset']
        Nt = self._metadata['Nt']

        return self._parser.read_array(offset, Nt)

    def get_bin_edges(self) -> np.ndarray:
        """Read bin edges array from file.

        Returns
        -------
        np.ndarray
            The bin edges array.
        """
        offset = self._metadata['bin_edges_offset']
        Nk = self._metadata['Nk']

        return self._parser.read_array(offset, Nk)

    def get_k_slice(self, k_index : int) -> np.ndarray:
        """Read a slice at k from data.

        Parameters
        ----------
        k_index : int
            The k index

        Returns
        -------
        np.ndarray
            The data at k vs tau.

        Raises
        ------
        IndexError
            If k_index is out of range.
        """
        # check index is in range
        Nk = self._metadata['Nk']
        if k_index < 0 or k_index >= Nk:
            raise IndexError(f'Index out of range. Choose an index between 0 and {Nk-1}.')

        offset = self._metadata['data_offset']
        Nt = self._metadata['Nt']
        Nextra = self._metadata['Nextra']
        offset += k_index * (Nt + Nextra) * calculate_format_size(self._parser.dtype)

        return self._parser.read_array(offset, Nt)
    
    def get_k_slice_err(self, k_index : int) -> np.ndarray:
        """Read a slice of uncertainty at k from data.

        Parameters
        ----------
        k_index : int
            The k index

        Returns
        -------
        np.ndarray
            The uncertainty of data at k vs tau.

        Raises
        ------
        IndexError
            If k_index is out of range.
        """
        # check index is in range
        Nk = self._metadata['Nk']
        if k_index < 0 or k_index >= Nk:
            raise IndexError(f'Index out of range. Choose an index between 0 and {Nk-1}.')

        offset = self._metadata['err_offset']
        Nt = self._metadata['Nt']
        Nextra = self._metadata['Nextra']
        offset += k_index * (Nt + Nextra) * calculate_format_size(self._parser.dtype)

        return self._parser.read_array(offset, Nt)


class AAParser(Parser):
    """Azimuthal average file parser class.
    Inherits from `Parser`. It adds the following unique methods:

    Methods
    -------
    read_metadata : dict
        Returns a dictionary containing the file metadata.
    """
    def __init__(self, fh : BinaryIO):
        super().__init__(fh)
        # check file identifier
        file_id = self._read_id()
        if file_id != 22:
            err_str = f'File identifier {file_id} not compatible with'
            err_str += ' azimuthal average file (22).'
            err_str += ' Input file might be wrong or corrupted.'
            raise RuntimeError(err_str)

    def read_metadata(self) -> dict:
        """Read metadata from the binary file.

        Returns
        -------
        dict
            The metadata dictionary.
        """
        metadata = {}

        # shape starts at byte 7
        # it comprises 4 values (Nt, Ny, Nx, Nextra), written as unsigned long long ('Q')
        metadata['Nk'] = self.read_value(7, 'Q')
        metadata['Nt'] = self.read_value(0, 'Q', whence=1)
        metadata['Nextra'] = self.read_value(0, 'Q', whence=1)

        # byte offsets start from end of file, written as unsigned long long ('Q')
        metadata['data_offset'] = self.read_value(-calculate_format_size('Q'), 'Q', 2)
        metadata['err_offset'] = self.read_value(-2 * calculate_format_size('Q'), 'Q', 1)
        metadata['k_offset'] = self.read_value(-2 * calculate_format_size('Q'), 'Q', 1)
        metadata['tau_offset'] = self.read_value(-2 * calculate_format_size('Q'), 'Q', 1)
        metadata['bin_edges_offset'] = self.read_value(-2 * calculate_format_size('Q'), 'Q', 1)

        return metadata


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
    err = np.zeros((len(fast.k), len(tau) + 2))

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
            err[i, :-2] = np.append(fast.err[i, :idx] * alpha, slow.err[i, Nt // 2 + 1:])

            # copy power spectrum and variance from slow
            data[i, -2:] = slow._data[i, -2:]
            err[i, -2:] = slow._err[i, -2:]

    return AzimuthalAverage(data, err, fast.k, tau, fast.bin_edges)


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
    data = np.zeros_like(az_avg1.data, shape=(dim_k, len(tau) + 2))
    err = np.zeros_like(az_avg1.err, shape=(dim_k, len(tau) + 2))

    # populate data
    data[:, :-2] = np.append(az_avg1.data, az_avg2.data, axis=1)[:,sortidx]
    err[:, :-2] = np.append(az_avg1.err, az_avg2.err, axis=1)[:,sortidx]

    # copy power spectrum and variance from input with longer tau
    if az_avg1.tau[-1] > az_avg2.tau[-1]:
        data[:,-2:] = az_avg1._data[:,-2:]
        err[:,-2:] = az_avg1._err[:,-2:]
    else:
        data[:,-2:] = az_avg2._data[:,-2:]
        err[:,-2:] = az_avg2._err[:,-2:]

    return AzimuthalAverage(data, err, az_avg1.k, tau[sortidx], az_avg1.bin_edges)
