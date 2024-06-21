# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

r"""This module contains the azimuthal average data class and the method to compute
the azimuthal average from the image structure function.

The azimuthal average can be computed from the image structure function as

.. code-block:: python

    import fastddm as fddm

    # compute image structure function dqt
    ...

    # compute azimuthal average masking out the central cross
    mask = fddm.mask.central_cross_mask(dqt.full_shape()[1:])
    aa = fddm.azimuthal_average(dqt, bins=bins, range=(0, dqt.ky[-1]), mask=mask)

The :py:class:`AzimuthalAverage` object is used to store and retrieve information
about the azimuthal average of the image structure function computed in DDM.

The :py:class:`AzimuthalAverage.data` contains the values of the azimuthal average of
the image structure function in :math:`(k, \Delta t)` order. For instance, the
azimuthal average at the 20th wave vector can be accessed via

.. code-block:: python

    aa.data[19]

.. note::
   Remember that Python uses zero-based indexing.

The :py:class:`AzimuthalAverage` can then be saved into a binary file by using
:py:meth:`AzimuthalAverage.save` (it will have a `.aa.ddm` extension)
and later retrieved from the memory using
:py:meth:`AAReader.load`, which you can call directly from ``fastddm`` as

.. code-block:: python

    # load image structure function
    dqt = fastddm.load('path/to/my_aa_file.aa.ddm')

Loading the :py:class:`AzimuthalAverage` from disk is not as demanding as for
the :py:class:`~fastddm.imagestructurefunction.ImageStructureFunction`.
However, also in this case we provide a fast reader through the
:py:class:`AAReader`, which can be used to access directly from the disk the
relevant data, for example:

.. code-block:: python

    from fastddm.azimuthalaverage import AAReader

    # open file
    r = AAReader('path/to/my_aa_file.aa.ddm')

    # access quantities
    # access tau array
    tau = r.get_tau()
    # access data for 20th k bin
    y = r.get_k_slice(k_index=19)
"""

from typing import Tuple, Optional, Union, Iterable, BinaryIO
from dataclasses import dataclass
import os
from sys import byteorder
import struct
import warnings
import numpy as np
from scipy.interpolate import interp1d

from .imagestructurefunction import ImageStructureFunction
from ._io_common import calculate_format_size, npdtype2format, Writer, Reader, Parser
from ._config import DTYPE


@dataclass
class AzimuthalAverage:
    """Azimuthal average container class.

    Parameters
    ----------
    _data : numpy.ndarray
        The packed data (azimuthal average of image structure function, power
        spectrum, and variance).
    _err : numpy.ndarray
        The packed uncertainties (uncertainty of the azimuthal average, power
        spectrum, and variance).
    k : numpy.ndarray
        The array of reference wavevector values in the bins.
    tau : numpy.ndarray
        The array of time delay values.
    bin_edges : numpy.ndarray
        The array of bin edges.
    """

    _data: np.ndarray
    _err: np.ndarray
    k: np.ndarray
    tau: np.ndarray
    bin_edges: np.ndarray

    @property
    def data(self) -> np.ndarray:
        """The azimuthal average of the 2D image structure function.

        Returns
        -------
        numpy.ndarray
            The azimuthal average data.
        """
        return self._data[:, :-2]

    @property
    def err(self) -> np.ndarray:
        """The uncertainty (standard deviation) in the azimuthal average
        of the 2D image structure function.

        Returns
        -------
        numpy.ndarray
            The uncertainty.
        """
        if self._err is None:
            return None
        else:
            return self._err[:, :-2]

    @property
    def power_spec(self) -> np.ndarray:
        """The azimuthal average of the average 2D power spectrum of the input
        images.

        Returns
        -------
        numpy.ndarray
            The azimuthal average of the power spectrum.
        """
        return self._data[:, -2]

    @property
    def var(self) -> np.ndarray:
        """The azimuthal average of the 2D variance (over time) of the Fourier
        transformed input images.

        Returns
        -------
        numpy.ndarray
            The azimuthal average of the variance.
        """
        return self._data[:, -1]

    @property
    def power_spec_err(self) -> np.ndarray:
        """The uncertainty in the azimuthal average of the average
        2D power spectrum of the input images.

        Returns
        -------
        numpy.ndarray
            The uncertainty in the azimuthal average of the power spectrum.
        """
        if self._err is None:
            return None
        else:
            return self._err[:, -2]

    @property
    def var_err(self) -> np.ndarray:
        """The uncertainty in the azimuthal average of the
        2D variance (over time) of the Fourier transformed input images.

        Returns
        -------
        numpy.ndarray
            The uncertainty in the azimuthal average of the variance.
        """
        if self._err is None:
            return None
        else:
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

    def save(self, fname: str = "analysis_blob") -> None:
        """Save ``AzimuthalAverage`` to binary file.

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

    def resample(self, tau: np.ndarray) -> "AzimuthalAverage":
        """Resample with new ``tau`` values and return a new ``AzimuthalAverage``.

        Parameters
        ----------
        tau : numpy.ndarray
            New values of ``tau``.

        Returns
        -------
        AzimuthalAverage
            The resampled azimuthal average.
        """
        # initialize data
        _data = np.zeros((len(self.k), len(tau) + 2), dtype=DTYPE)
        is_err = self._err is not None
        if is_err:
            _err = np.zeros((len(self.k), len(tau) + 2), dtype=DTYPE)
        else:
            _err = None

        _tau = np.log(tau)

        # loop through k values
        for i in range(len(self.k)):
            # check for nan
            if np.isnan(self.data[i, 0]):
                _data[i, :-2] = np.full(len(tau), np.nan, dtype=DTYPE)
                if is_err:
                    _err[i, :-2] = np.full(len(tau), np.nan, dtype=DTYPE)
            else:
                # interpolate points in loglog scale
                f = interp1d(
                    x=np.log(self.tau),
                    y=np.log(self.data[i]),
                    kind="quadratic",
                    fill_value="extrapolate",
                )
                _data[i, :-2] = np.exp(f(_tau))

                if is_err:
                    # interpolate uncertainties in loglog scale
                    f = interp1d(
                        x=np.log(self.tau),
                        y=np.log(self.err[i]),
                        kind="quadratic",
                        fill_value="extrapolate",
                    )
                    _err[i, :-2] = np.exp(f(_tau))

        # append power_spec and var
        _data[:, -2] = self.power_spec
        _data[:, -1] = self.var
        if is_err:
            _err[:, -2] = self.power_spec_err
            _err[:, -1] = self.var_err

        # ensure DTYPE for all AzimuthalAverage args
        k = self.k.astype(DTYPE)
        bin_edges = self.bin_edges.astype(DTYPE)

        return AzimuthalAverage(_data, _err, k, tau.astype(DTYPE), bin_edges)


def azimuthal_average_array(
    data: np.ndarray,
    dist: np.ndarray,
    bins: Optional[Union[int, Iterable[float]]] = 10,
    range: Optional[Tuple[float, float]] = None,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    counts: Optional[np.ndarray] = None,
    eval_err: Optional[bool] = True,
) -> Tuple[np.ndarray, ...]:
    r"""Compute the azimuthal average of a 3D array.

    For every bin :math:`k`, the average is calculated as

    .. math::

        \bar{x}_k = \frac{\sum_i w_i x_i}{\sum_i w_i} ,

    where :math:`w_i` is the weight given to the pixel :math:`i`.
    The sum runs over the elements :math:`i \in \mathcal{S}_k`, where
    :math:`\mathcal{S}_k` is the subset of elements :math:`i` with distance
    ``dist`` in the bin.
    The ``mask`` allows to exclude certain pixels from the calculation.
    A pixel :math:`i` is counted as many times as indicated by ``counts``.
    The uncertainty is calculated as the square root of the unbiased variance
    for weighed measures

    .. math::

        \text{VAR}(x_k) = \frac{\sum_i w_i (x_i - \bar{x}_k)^2}{\sum_i w_i - \sum_i w_i^2 / \sum_i w_i}.

    Parameters
    ----------
    data : numpy.ndarray
        The 3D input array.
    dist : numpy.ndarray
        A 2D array storing the distances from a center. The array must have the
        same y, x shape as ``data``.
    bins : Union[int, Iterable[float]], optional
        If ``bins`` is an int, it defines the number of equal-width bins in the
        given range (10, by default). If ``bins`` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(dist.min(), dist.max())``. Values outside the range are ignored.
    mask : numpy.ndarray, optional
        If a boolean ``mask`` is given, it is used to exclude points from
        the azimuthal average (where False is set). The array must have the
        same shape of ``dist``. If ``mask`` is not of boolean type, it is cast
        and a warning is raised.
    weights : numpy.ndarray, optional
        An array of non-negative weights, of the same shape as ``dist``. Each
        value in ``data`` and ``dist`` only contributes its associated weight
        (instead of 1).
    counts : numpy.ndarray, optional
        An array of bin counts, of the same shape as ``dist``.
        Each value in ``data`` and ``dist`` is sampled its associated number of
        counts (instead of 1).
    eval_err : bool, optional
        If True, the uncertainty is computed. Default is True.

    Returns
    -------
    Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The azimuthal average, its uncertainty, the distances, and the bin
        edges.

    Raises
    ------
    ValueError
        If ``dist``, ``mask``, ``weights``, or ``counts`` are not compatible
        with shape of ``data``.
    """
    # read data shape
    dim_y, dim_x = data.shape[-2:]

    # check dist
    if dist.shape != (dim_y, dim_x):
        raise ValueError("dist shape must be compatible with data shape.")

    # check mask
    if mask is not None:
        _mask = mask.copy()
        if mask.shape != dist.shape:
            raise ValueError("mask shape and dist shape must be the same.")
        if mask.dtype != bool:
            _mask = mask.astype(bool)
            warnings.warn("mask is not of boolean type, it is cast to bool.")
    else:
        _mask = np.full((dist.shape), True)

    # check weights
    if weights is not None:
        if weights.shape != dist.shape:
            raise ValueError("weights shape and dist shape must be the same.")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative.")

    # check counts
    if counts is not None:
        if counts.shape != dist.shape:
            raise ValueError("counts shape and dist shape must be the same.")
    else:
        counts = np.ones(dist.shape, dtype=DTYPE)

    # check range
    if range is None:
        x_min = np.min(dist)
        x_max = np.max(dist)
    else:
        x_min, x_max = range
        if x_min > x_max:
            x_min, x_max = x_max, x_min

    # compute bin edges
    if isinstance(bins, int):
        bin_edges = np.linspace(x_min, x_max, bins, dtype=DTYPE)
        n_bins = bins
    elif isinstance(bins, Iterable):
        # check if it is a monotonically increasing array
        if not np.all(bins[1:] >= bins[:-1]):
            raise ValueError("bins must be monotonically increasing.")
        bin_edges = np.array(bins).astype(DTYPE)
        n_bins = len(bins)
        x_min = bins[0]
    else:
        raise ValueError("bins must be an int or an iterable.")

    # digitize dist
    bin_indices = np.searchsorted(bin_edges, dist)

    # update mask
    _mask[(bin_indices == n_bins) | (dist < x_min)] = False

    # correct weights for counts
    wi_corr = counts.astype(DTYPE)
    if weights is not None:
        wi_corr *= weights

    # initialize outputs
    x = np.zeros(n_bins, dtype=DTYPE)
    avg = np.zeros((n_bins, len(data)), dtype=DTYPE)
    err = None
    if eval_err:
        err = np.zeros((n_bins, len(data)), dtype=DTYPE)

        # compute squared weights and correct for counts
        wi2_corr = counts.astype(DTYPE)
        if weights is not None:
            wi2_corr *= weights**2

    # update mask
    _mask[np.logical_not(wi_corr)] = False

    # calculate sum of weights
    sum_wi = np.zeros(n_bins, dtype=DTYPE)
    sum_wi2 = np.zeros(n_bins, dtype=DTYPE)
    for (i, j), bin_idx in np.ndenumerate(bin_indices):
        if _mask[i, j]:
            sum_wi[bin_idx] += wi_corr[i, j]
            sum_wi2[bin_idx] += wi2_corr[i, j]

    # calculate the azimuthal average
    for (i, j), bin_idx in np.ndenumerate(bin_indices):
        if _mask[i, j]:
            x[bin_idx] += dist[i, j] * wi_corr[i, j] / sum_wi[bin_idx]
            avg[bin_idx] += data[:, i, j] * wi_corr[i, j] / sum_wi[bin_idx]
    # replace missing values from average with nan
    avg[sum_wi == 0] = np.nan

    # replace values in k where sum_wi is 0 with bin edges mid points
    for bin_idx in np.arange(n_bins):
        if sum_wi[bin_idx] == 0:
            if bin_idx > 0:
                x[bin_idx] = 0.5 * (bin_edges[bin_idx - 1] + bin_edges[bin_idx])
            else:
                x[0] = bin_edges[0]

    # calculate the uncertainty
    if eval_err:
        # compute the bias factor
        bias_factor = [
            swi - (swi2 / swi) if swi > 0 else 0 for (swi, swi2) in zip(sum_wi, sum_wi2)
        ]

        # compute variance
        with np.errstate(divide="ignore", invalid="ignore"):
            for (i, j), bin_idx in np.ndenumerate(bin_indices):
                if _mask[i, j]:
                    err[bin_idx] += (
                        wi_corr[i, j]
                        * (data[:, i, j] - avg[bin_idx]) ** 2
                        / bias_factor[bin_idx]
                    )
        # replace missing values with nan
        err[sum_wi == 0] = np.nan

        # take square root
        np.sqrt(err, out=err)

    return avg, err, x, bin_edges


def azimuthal_average(
    img_str_func: ImageStructureFunction,
    bins: Optional[Union[int, Iterable[float]]] = 10,
    range: Optional[Tuple[float, float]] = None,
    mask: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    eval_err: Optional[bool] = True,
) -> AzimuthalAverage:
    r"""Compute the azimuthal average of the image structure function.

    For every bin :math:`i`, the average is calculated as

    .. math::

        \bar{x}_i = \frac{\sum_k w_k x_k}{\sum_k w_k} ,

    where :math:`w_k` is the weight given to the wavevector :math:`k`.
    The sum runs over the elements :math:`k \in \mathcal{S}_i`, where
    :math:`\mathcal{S}_i` is the subset of wavevector :math:`k` with modulus
    in the bin :math:`i`.
    The ``mask`` allows to exclude certain wavevectors from the calculation.
    The uncertainty is calculated as the square root of the unbiased variance
    for weighed measures

    .. math::

        \text{VAR}(x_i) = \frac{\sum_k w_k (x_k - \bar{x}_i)^2}{\sum_k w_k - \sum_k w_k^2 / \sum_k w_k}.

    Parameters
    ----------
    img_str_func : ImageStructureFunction
        The image structure function.
    bins : Union[int, Iterable[float]], optional
        If ``bins`` is an int, it defines the number of equal-width bins in the
        given range (10, by default). If ``bins`` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    range : Tuple[float, float], optional
        The lower and upper range of the bins. If not provided, range is simply
        ``(k.min(), k.max())``, where ``k`` is the vector modulus computed from
        ``kx`` and ``ky``. Values outside the range are ignored. The first
        element of the range must be less than or equal to the second.
    mask : numpy.ndarray, optional
        If a boolean ``mask`` is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same ``(y, x)`` shape of ``data``. If ``mask`` is not of boolean type,
        it is cast to booland a ``warning`` is raised.
    weights : numpy.ndarray, optional
        An array of weights, of the same ``(y, x)`` shape as ``data``. Each
        value in ``data`` only contributes its associated weight towards
        the bin count (instead of 1).
    eval_err : bool, optional
        If True, the uncertainty is computed. Default is True.

    Returns
    -------
    AzimuthalAverage
        The azimuthal average.
    """
    # get the counts
    # the first column is counted once
    # the last column is counted once if the full width is even
    # the other elements are counted twice
    counts = np.full(img_str_func.shape[1:], 2)
    counts[:, 0] = 1
    counts[:, -1] = 1 if img_str_func.width % 2 == 0 else 2

    # compute the k modulus
    X, Y = np.meshgrid(img_str_func.kx, img_str_func.ky)
    k_modulus = np.sqrt(X**2 + Y**2)

    # compute the azimuthal average
    az_avg, err, k, bin_edges = azimuthal_average_array(
        img_str_func._data,
        k_modulus,
        bins,
        range,
        mask,
        weights,
        counts,
        eval_err,
    )

    return AzimuthalAverage(az_avg, err, k, img_str_func.tau, bin_edges)


class AAWriter(Writer):
    """Azimuthal average writer class. Inherits from ``Writer``.

    It adds the unique method ``write_obj``.

    Defines the functions to write :py:class:`AzimuthalAverage` object to binary file.

    The structure of the binary file is the following:

    Header:

    * bytes 0-1: endianness, string, utf-8 encoding [``"LL"`` = 'little', ``"BB"`` = 'big']
    * bytes 2-3: file identifier, 16-bit integer, unsigned short [``22`` for azimuthal average]
    * bytes 4-5: file version, pair of 8-bit integers as (major_version, minor_version), unsigned char
    * byte 6: dtype, string, utf-8 encoding [``"d"`` = float64, ``"f"`` = float32]
    * bytes 7-14: data height, 64-bit integer, unsigned long long
    * bytes 15-22: data width, 64-bit integer, unsigned long long
    * bytes 23-30: extra slices, 64-bit integer, unsigned long long
    * byte 31: flag for standard deviation of data, 8-bit integer, unsigned char [``0`` if ``err`` is None, ``1`` if it is stored in the dataclass]

    The data is stored in 'C' order and ``dtype`` format as follows:

    * from byte ``data_offset``: ``_data``
    * from byte ``err_offset``: ``_err``
    * from byte ``k_offset``: ``k`` array
    * from byte ``tau_offset``: ``tau`` array
    * from byte ``bin_edges_offset``: ``bin_edges`` array

    From the end of the file,
    the byte offsets are stored as unsigned long long 64-bit integers in this order:

    * ``data_offset``
    * ``err_offset``
    * ``k_offset``
    * ``tau_offset``
    * ``bin_edges_offset``
    """

    def write_obj(self, obj: AzimuthalAverage) -> None:
        """Write ``AzimuthalAverage`` object to binary file.

        Parameters
        ----------
        obj : AzimuthalAverage
            ``AzimuthalAverage`` object.
        """
        # get data dtype
        dtype = npdtype2format(obj.data.dtype.name)
        # get data shape
        Nk, Nt = obj.shape
        # assign Nextra = 2
        # we have power_spectrum + variance
        Nextra = obj._data.shape[-1] - Nt
        # get if error is present
        is_err = obj._err is not None

        # write file header
        self._write_header(Nk, Nt, Nextra, is_err, dtype)

        # write data
        self._write_data(obj)

    def _write_header(
        self, Nk: int, Nt: int, Nextra: int, is_err: bool, dtype: str
    ) -> None:
        """Write azimuthal average file header.

        In version 0.2, the header is structured as follows:
        * bytes 0-1: endianness (`LL` = 'little'; `BB` = 'big'), 'utf-8' encoding
        * bytes 2-3: file identifier (22), `H` (unsigned short)
        * bytes 4-5: file version as (major_version, minor_version), `BB` (unsigned char)
        * byte 6: dtype (`d` = float64; `f` = float32), 'utf-8' encoding
        * bytes 7-14: data height, `Q` (unsigned long long)
        * bytes 15-22: data width, `Q` (unsigned long long)
        * bytes 23-30: extra slices, `Q` (unsigned long long)
        * byte 31: 0 if error is None, 1 otherwise, `B` (unsigned char)

        Parameters
        ----------
        Nk : int
            Height.
        Nt : int
            Width.
        Nextra : int
            Number of extra slices.
        is_err : bool
            True if error is computed.
        dtype : str
            Data dtype.
        """
        curr_byte_len = 0
        # system endianness
        if byteorder == "little":
            self._fh.write("LL".encode("utf-8"))
        else:
            self._fh.write("BB".encode("utf-8"))
        curr_byte_len += 2

        # file identifier
        self._fh.write(struct.pack("H", 22))
        curr_byte_len += calculate_format_size("H")

        # file version
        self._fh.write(struct.pack("BB", *(self.version)))
        curr_byte_len += 2 * calculate_format_size("B")

        # dtype
        self._fh.write(dtype.encode("utf-8"))
        curr_byte_len += 1

        # height
        self._fh.write(struct.pack("Q", Nk))
        curr_byte_len += calculate_format_size("Q")

        # width
        self._fh.write(struct.pack("Q", Nt))
        curr_byte_len += calculate_format_size("Q")

        # extra slices
        self._fh.write(struct.pack("Q", Nextra))
        curr_byte_len += calculate_format_size("Q")

        # is error
        self._fh.write(struct.pack("B", 1 if is_err else 0))
        curr_byte_len += calculate_format_size("B")

        # add empty bytes up to HEAD_BYTE_LEN for future use (if needed)
        self._fh.write(bytearray(self.head_byte_len - curr_byte_len))

    def _write_data(self, obj: AzimuthalAverage) -> None:
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
        Nt = len(obj.tau)
        Nk, dim_t = obj._data.shape
        Nextra = dim_t - Nt

        # write _data
        obj._data.tofile(self._fh)
        data_offset = self.head_byte_len

        # write _err
        err_offset = data_offset + Nk * (Nt + Nextra) * calculate_format_size(fmt)
        if obj._err is not None:
            obj._err.tofile(self._fh)

        # write k, tau, and bin_edges
        obj.k.tofile(self._fh)
        obj.tau.tofile(self._fh)
        obj.bin_edges.tofile(self._fh)
        if obj._err is not None:
            k_offset = err_offset + Nk * (Nt + Nextra) * calculate_format_size(fmt)
        else:
            k_offset = err_offset
        tau_offset = k_offset + Nk * calculate_format_size(fmt)
        bin_edges_offset = tau_offset + Nt * calculate_format_size(fmt)

        # write byte offsets
        self._fh.write(struct.pack("Q", bin_edges_offset))
        self._fh.write(struct.pack("Q", tau_offset))
        self._fh.write(struct.pack("Q", k_offset))
        self._fh.write(struct.pack("Q", err_offset))
        self._fh.write(struct.pack("Q", data_offset))


class AAReader(Reader):
    """FastDDM azimuthal average reader class.
    Inherits from ``Reader``.
    """

    def __init__(self, file: str):
        super().__init__(file)
        self._parser = AAParser(self._fh)
        self._metadata = self._parser.read_metadata()

    def load(self) -> AzimuthalAverage:
        """Load azimuthal average from file.

        Returns
        -------
        AzimuthalAverage
            The ``AzimuthalAverage`` object.

        Raises
        ------
        IOError
            If file version not supported.
        """
        # check version supported
        if not self._parser.supported:
            version = self._parser.get_version()
            raise IOError(f"File version {version} not supported.")

        # get data shape
        Nk = self._metadata["Nk"]
        Nt = self._metadata["Nt"]
        Nextra = self._metadata["Nextra"]
        shape = (Nk, Nt + Nextra)

        if self._metadata["is_err"]:
            return AzimuthalAverage(
                self._parser.read_array(self._metadata["data_offset"], shape),
                self._parser.read_array(self._metadata["err_offset"], shape),
                self.get_k(),
                self.get_tau(),
                self.get_bin_edges(),
            )
        else:
            return AzimuthalAverage(
                self._parser.read_array(self._metadata["data_offset"], shape),
                None,
                self.get_k(),
                self.get_tau(),
                self.get_bin_edges(),
            )

    def get_k(self) -> np.ndarray:
        """Read ``k`` array from file.

        Returns
        -------
        numpy.ndarray
            The ``k`` array.
        """
        offset = self._metadata["k_offset"]
        Nk = self._metadata["Nk"]

        return self._parser.read_array(offset, Nk)

    def get_tau(self) -> np.ndarray:
        """Read ``tau`` array from file.

        Returns
        -------
        numpy.ndarray
            The ``tau`` array.
        """
        offset = self._metadata["tau_offset"]
        Nt = self._metadata["Nt"]

        return self._parser.read_array(offset, Nt)

    def get_bin_edges(self) -> np.ndarray:
        """Read ``bin_edges`` array from file.

        Returns
        -------
        numpy.ndarray
            The ``bin_edges`` array.
        """
        offset = self._metadata["bin_edges_offset"]
        Nk = self._metadata["Nk"]

        return self._parser.read_array(offset, Nk)

    def get_k_slice(self, k_index: int) -> np.ndarray:
        """Read a slice at ``k_index`` from data.

        Parameters
        ----------
        k_index : int
            The ``k`` index

        Returns
        -------
        numpy.ndarray
            The ``data`` at ``k`` vs ``tau``.

        Raises
        ------
        IndexError
            If ``k_index`` is out of range.
        """
        # check index is in range
        Nk = self._metadata["Nk"]
        if k_index < 0 or k_index >= Nk:
            raise IndexError(
                f"Index out of range. Choose an index between 0 and {Nk-1}."
            )

        offset = self._metadata["data_offset"]
        Nt = self._metadata["Nt"]
        Nextra = self._metadata["Nextra"]
        offset += k_index * (Nt + Nextra) * calculate_format_size(self._parser.dtype)

        return self._parser.read_array(offset, Nt)

    def get_k_slice_err(self, k_index: int) -> np.ndarray:
        """Read a slice of uncertainty at ``k_index`` from ``data``.

        Parameters
        ----------
        k_index : int
            The ``k`` index

        Returns
        -------
        numpy.ndarray
            The uncertainty of ``data`` at ``k`` vs ``tau``.

        Raises
        ------
        IndexError
            If ``k_index`` is out of range.
        """
        # check index is in range
        Nk = self._metadata["Nk"]
        if k_index < 0 or k_index >= Nk:
            raise IndexError(
                f"Index out of range. Choose an index between 0 and {Nk-1}."
            )

        offset = self._metadata["err_offset"]
        Nt = self._metadata["Nt"]
        Nextra = self._metadata["Nextra"]
        offset += k_index * (Nt + Nextra) * calculate_format_size(self._parser.dtype)

        if self._metadata["is_err"]:
            return self._parser.read_array(offset, Nt)
        else:
            return None


class AAParser(Parser):
    """Azimuthal average file parser class.
    Inherits from ``Parser``.
    """

    def __init__(self, fh: BinaryIO):
        super().__init__(fh)
        # check file identifier
        file_id = self._read_id()
        if file_id != 22:
            err_str = f"File identifier {file_id} not compatible with"
            err_str += " azimuthal average file (22)."
            err_str += " Input file might be wrong or corrupted."
            raise RuntimeError(err_str)

    def read_metadata(self) -> dict:
        """Read metadata from the binary file.

        Returns
        -------
        dict
            The metadata dictionary.
        """
        metadata = {}
        version = self.get_version()

        # shape starts at byte 7
        # it comprises 4 values (Nt, Ny, Nx, Nextra), written as unsigned long long ('Q')
        # plus a value for the presence of the uncertainty (is_err), written as unsigned char ('B')
        metadata["Nk"] = self.read_value(7, "Q")
        metadata["Nt"] = self.read_value(0, "Q", whence=1)
        metadata["Nextra"] = self.read_value(0, "Q", whence=1)
        if version > (0, 1):
            metadata["is_err"] = bool(self.read_value(0, "B", whence=1))
        else:
            metadata["is_err"] = False

        # byte offsets start from end of file, written as unsigned long long ('Q')
        metadata["data_offset"] = self.read_value(
            -calculate_format_size("Q"), "Q", whence=2
        )
        if version > (0, 1):
            metadata["err_offset"] = self.read_value(
                -2 * calculate_format_size("Q"), "Q", whence=1
            )
        else:
            metadata["err_offset"] = 0
        metadata["k_offset"] = self.read_value(
            -2 * calculate_format_size("Q"), "Q", whence=1
        )
        metadata["tau_offset"] = self.read_value(
            -2 * calculate_format_size("Q"), "Q", whence=1
        )
        metadata["bin_edges_offset"] = self.read_value(
            -2 * calculate_format_size("Q"), "Q", whence=1
        )

        return metadata


def melt(az_avg1: AzimuthalAverage, az_avg2: AzimuthalAverage) -> AzimuthalAverage:
    """Melt two azimuthal averages into one object.

    The melt is performed as follows:

    * the "slow" acquisition is identified as the ``az_avg`` having the largest
      ``.tau[0]`` value
    * the first 10 data points are taken from the "slow" ``az_avg``
    * data points from the "fast" ``az_avg`` at the time delays of the first
      10 ``tau`` of the "slow" ``az_avg`` are obtained via cubic
      interpolation of the log-log scaled fast data
    * a multiplicative correction factor is obtained via least squares
      minimization and the "fast" data points are scaled onto the "slow" ones

    The ``var`` and ``power_spec`` are taken from the "slow" ``az_avg``.

    Parameters
    ----------
    az_avg1 : AzimuthalAverage
        One :py:class:`AzimuthalAverage` object.
    az_avg2 : AzimuthalAverage
        Another :py:class:`AzimuthalAverage` object.

    Returns
    -------
    AzimuthalAverage
        The two :py:class:`AzimuthalAverage` objects, merged into a new one.
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
    tau = np.append(fast.tau[:idx], slow.tau[Nt // 2 + 1 :]).astype(DTYPE)
    data = np.zeros((len(fast.k), len(tau) + 2), dtype=DTYPE)
    if fast._err is None or slow._err is None:
        err = None
    else:
        err = np.zeros((len(fast.k), len(tau) + 2), dtype=DTYPE)

    t = np.log(slow.tau[:Nt])

    # loop through k values
    for i in range(len(fast.k)):
        # check for nan
        if np.any(np.isnan([fast.data[i, 0], slow.data[i, 0]])):
            data[i] = np.full(len(tau) + 2, np.nan, dtype=DTYPE)
        else:
            # find multiplicative factor via least squares minimization
            # interpolate in loglog scale (smoother curve)
            f = interp1d(x=np.log(fast.tau), y=np.log(fast.data[i]), kind="cubic")
            sum_xiyi = np.sum(np.exp(f(t)) * slow.data[i, :Nt])
            sum_xi2 = np.sum(np.exp(f(t)) ** 2)
            alpha = sum_xiyi / sum_xi2

            # scale fast on slow
            data[i, :-2] = np.append(
                fast.data[i, :idx] * alpha, slow.data[i, Nt // 2 + 1 :]
            ).astype(DTYPE)
            if err is not None:
                err[i, :-2] = np.append(
                    fast.err[i, :idx] * alpha, slow.err[i, Nt // 2 + 1 :]
                ).astype(DTYPE)

            # copy power spectrum and variance from slow
            data[i, -2:] = slow._data[i, -2:]
            if err is not None:
                err[i, -2:] = slow._err[i, -2:]

    # ensure DTYPE for all AzimuthalAverage args
    k = fast.k.astype(DTYPE)
    bin_edges = fast.bin_edges.astype(DTYPE)

    return AzimuthalAverage(data, err, k, tau, bin_edges)


def mergesort(az_avg1: AzimuthalAverage, az_avg2: AzimuthalAverage) -> AzimuthalAverage:
    """Merge the values of two azimuthal averages.

    Values will then be sorted based on ``tau``.

    Parameters
    ----------
    az_avg1 : AzimuthalAverage
        One :py:class:`AzimuthalAverage` object.
    az_avg2 : AzimuthalAverage
        Another :py:class:`AzimuthalAverage` object.

    Returns
    -------
    AzimuthalAverage
        The two :py:class:`AzimuthalAverage` objects are fused into a new one.
    """
    tau = np.append(az_avg1.tau, az_avg2.tau).astype(DTYPE)
    sortidx = np.argsort(tau)

    # create new data
    dim_k, dim_tau = az_avg1.shape
    data = np.zeros(shape=(dim_k, len(tau) + 2), dtype=DTYPE)
    if az_avg1._err is None or az_avg2._err is None:
        err = None
    else:
        err = np.zeros(shape=(dim_k, len(tau) + 2), dtype=DTYPE)

    # populate data
    data[:, :-2] = np.append(az_avg1.data, az_avg2.data, axis=1)[:, sortidx].astype(
        DTYPE
    )
    if err is not None:
        err[:, :-2] = np.append(az_avg1.err, az_avg2.err, axis=1)[:, sortidx].astype(
            DTYPE
        )

    # copy power spectrum and variance from input with longer tau
    if az_avg1.tau[-1] > az_avg2.tau[-1]:
        data[:, -2:] = az_avg1._data[:, -2:]
        if err is not None:
            err[:, -2:] = az_avg1._err[:, -2:]
    else:
        data[:, -2:] = az_avg2._data[:, -2:]
        if err is not None:
            err[:, -2:] = az_avg2._err[:, -2:]

    # ensure DTYPE for all AzimuthalAverage args
    k = az_avg1.k.astype(DTYPE)
    bin_edges = az_avg1.bin_edges.astype(DTYPE)

    return AzimuthalAverage(data, err, k, tau[sortidx], bin_edges)
