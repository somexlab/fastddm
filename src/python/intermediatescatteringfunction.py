# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

r"""This module contains the intermediate scattering function data class, and auxiliary classes &
methods.

The :py:class:`IntermediateScatteringFunction` object is used to store and retrieve information
about the intermediate scattering function (ISF). The ISF can be estimated from the azimuthally
averaged Image Structure Function, if some assumptions about the functional shape of the Image
Structure Function are made. The method :py:func:`intermediatescatteringfunction.azavg2isf_estimate`
provides this functionality, by assuming a basic functional shape of the image structure function
:math:`D(q,\Delta t) = A(q)\left[ 1 - ISF(q, \Delta t) \right] + B(q)`. For more information see
the docstring.

The :py:class:`IntermediateScatteringFunction.data` contains the ISF values in
:math:`(k, \Delta t)` order. For instance, the ISF computed at the 10th k vector bin can be accessed via

.. code-block:: python

    isf.data[9]

.. note::
   Remember that Python uses zero-based indexing.

The :py:class:`IntermediateScatteringFunction` can then be saved into a binary file by using
:py:meth:`IntermediateScatteringFunction.save` (by default called ``analysis_blob``, with the
extension ``.isf.ddm``) and later retrieved from the memory using
:py:meth:`ISFReader.load`, which you can call directly from ``fastddm`` as

.. code-block:: python

    # load intermediate scattering function
    isf = fastddm.load('path/to/my_isf_file.isf.ddm')

In order to avoid reading and loading the entire file, we also provide
a fast reader through the :py:class:`ISFReader`, which can be used to
access directly from the disk the relevant data, for example:

.. code-block:: python

    from fastddm.intermediatescatteringfunction import ISFReader

    # open file
    r = ISFReader('path/to/my_isf_file.isf.ddm')

    # access quantities
    # access k array
    k = r.get_k()

    # access delays/lags
    dt = r.get_tau()

    # access a slice of the ISF for the sixth k value
    isf_slice = r.get_k_slice(k_index=5)
"""

import os
import struct
from dataclasses import dataclass
from sys import byteorder
from typing import BinaryIO, Optional, Tuple

import numpy as np
from scipy.interpolate import interp1d

from ._config import DTYPE
from ._io_common import Parser, Reader, Writer, calculate_format_size, npdtype2format
from .azimuthalaverage import AzimuthalAverage
from .noise_est import estimate_camera_noise


@dataclass
class IntermediateScatteringFunction:
    """Intermediate scattering function container class.

    Parameters
    ----------
    _data : numpy.ndarray
        The data (intermediate scattering function).
    _err : Optional[numpy.ndarray]
        The uncertainties (uncertainty of the intermediate scattering
        function).
    k : numpy.ndarray
        The array of reference wavevector values in the bins.
    tau : numpy.ndarray
        The array of time delay values.
    bin_edges : numpy.ndarray
        The array of bin edges (from the azimuthal average of the image
        structure function).
    """

    _data: np.ndarray
    _err: Optional[np.ndarray]
    k: np.ndarray
    tau: np.ndarray
    bin_edges: np.ndarray

    @property
    def data(self) -> np.ndarray:
        """The intermediate scattering function.

        Returns
        -------
        numpy.ndarray
            The intermediate scattering function data.
        """
        return self._data

    @property
    def err(self) -> Optional[np.ndarray]:
        """The uncertainty (standard deviation) in the intermediate scattering
        function.

        Returns
        -------
        numpy.ndarray
            The uncertainty.
        """
        if self._err is None:
            return None
        else:
            return self._err

    @property
    def shape(self) -> Tuple[int, int]:
        """The shape of the azimuthal average data.

        Returns
        -------
        Tuple[int, int]
            The shape of the data.
        """
        return self.data.shape  # type: ignore

    def save(self, fname: str = "analysis_blob") -> None:
        """Save IntermediateScatteringFunction to binary file.

        Parameters
        ----------
        fname : str, optional
            The full file name, by default "analysis_blob".
        """
        # check name
        dir, name = os.path.split(fname)
        name = name if name.endswith(".isf.ddm") else f"{name}.isf.ddm"

        # save to file
        with ISFWriter(file=os.path.join(dir, name)) as f:
            f.write_obj(self)

    def resample(self, tau: np.ndarray) -> "IntermediateScatteringFunction":
        """Resample with new tau values and return a new
        IntermediateScatteringFunction.

        Parameters
        ----------
        tau : numpy.ndarray
            New values of tau.

        Returns
        -------
        IntermediateScatteringFunction
            A new instance of the resampled intermediate scattering function.
        """
        # initialize data
        _data = np.zeros((len(self.k), len(tau)), dtype=DTYPE)
        is_err = self._err is not None
        if is_err:
            _err = np.zeros((len(self.k), len(tau)), dtype=DTYPE)
        else:
            _err = None

        _tau = np.log(tau)

        # loop through k values
        for i in range(len(self.k)):
            # check for nan
            if np.isnan(self.data[i, 0]):
                _data[i] = np.full(len(tau), np.nan, dtype=DTYPE)
                if is_err and _err is not None:
                    _err[i] = np.full(len(tau), np.nan, dtype=DTYPE)
            else:
                # interpolate points in loglog scale
                f = interp1d(
                    x=np.log(self.tau),
                    y=np.log(self.data[i]),
                    kind="quadratic",
                    fill_value="extrapolate",  # type: ignore
                )
                _data[i] = np.exp(f(_tau))

                if is_err and _err is not None and self.err is not None:
                    # interpolate uncertainties in loglog scale
                    f = interp1d(
                        x=np.log(self.tau),
                        y=np.log(self.err[i]),
                        kind="quadratic",
                        fill_value="extrapolate",  # type: ignore
                    )
                    _err[i] = np.exp(f(_tau))

        # ensure DTYPE for all IntermediateScatteringFunction args
        k = self.k.astype(DTYPE)
        bin_edges = self.bin_edges.astype(DTYPE)

        return IntermediateScatteringFunction(
            _data, _err, k, tau.astype(DTYPE), bin_edges
        )


class ISFWriter(Writer):
    """Intermediate scattering function writer class. Inherits from ``Writer``.

    It adds the unique method ``write_obj``.

    Defines the functions to write :py:class:`IntermediateScatteringFunction` object to binary file.

    The structure of the binary file is the following:

    Header:

    * bytes 0-1: endianness, string, utf-8 encoding [``"LL"`` = 'little'; ``"BB"`` = 'big']
    * bytes 2-3: file identifier, 16-bit integer, unsigned short [``43`` for intermediate scattering function]
    * bytes 4-5: file version, pair of 8-bit integers as (major_version, minor_version), unsigned char
    * byte 6: dtype, string, utf-8 encoding [``"d"`` = float64, ``"f"`` = float32]
    * bytes 7-14: data height, 64-bit integer, unsigned long long
    * bytes 15-22: data width, 64-bit integer, unsigned long long
    * bytes 23-30: extra slices, 64-bit integer, unsigned long long
    * byte 31: flag for standard deviation of data, 8-bit integer, unsigned char [``0`` if ``err`` is None, ``1`` if it is stored in the dataclass]

    The data is stored in 'C' order and `dtype` format as follows:

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

    def write_obj(self, obj: IntermediateScatteringFunction) -> None:
        """Write IntermediateScatteringFunction object to binary file.

        Parameters
        ----------
        obj : IntermediateScatteringFunction
            IntermediateScatteringFunction object.
        """
        # get data dtype
        dtype = npdtype2format(obj.data.dtype.name)
        # get data shape
        Nk, Nt = obj.shape
        # assign Nextra = 0
        Nextra = 0
        # get if error is present
        is_err = obj._err is not None

        # write file header
        self._write_header(Nk, Nt, Nextra, is_err, dtype)

        # write data
        self._write_data(obj)

    def _write_header(
        self, Nk: int, Nt: int, Nextra: int, is_err: bool, dtype: str
    ) -> None:
        """Write intermediate scattering function file header.

        In version 0.2, the header is structured as follows:
        * bytes 0-1: endianness (`LL` = 'little'; `BB` = 'big'), 'utf-8' encoding
        * bytes 2-3: file identifier (43), `H` (unsigned short)
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
        self._fh.write(struct.pack("H", 43))
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

    def _write_data(self, obj: IntermediateScatteringFunction) -> None:
        """Write intermediate scattering function data.

        In version 0.1, the data is stored in 'C' order and `dtype` format as
        follows:
        * from `data_offset`: _data
        * from `err_offset`: _err
        * from `k_offset`: `k` array
        * from `tau_offset`: `tau` array
        * from `bin_edges_offset`: `bin_edges` array

        From the end of the file,
        the byte offsets are stored in `Q` (unsigned long long) format in this
        order:
        * `data_offset`
        * `err_offset`
        * `k_offset`
        * `tau_offset`
        * `bin_edges_offset`

        Parameters
        ----------
        obj : IntermediateScatteringFunction
            The intermediate scattering function object.
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


class ISFReader(Reader):
    """FastDDM intermediate scattering function reader class.

    Inherits from `Reader`.
    """

    def __init__(self, file: str):
        super().__init__(file)
        self._parser = ISFParser(self._fh)
        self._metadata = self._parser.read_metadata()

    def load(self) -> IntermediateScatteringFunction:
        """Load intermediate scattering function from file.

        Returns
        -------
        IntermediateScatteringFunction
            The IntermediateScatteringFunction object.

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
            return IntermediateScatteringFunction(
                self._parser.read_array(self._metadata["data_offset"], shape),
                self._parser.read_array(self._metadata["err_offset"], shape),
                self.get_k(),
                self.get_tau(),
                self.get_bin_edges(),
            )
        else:
            return IntermediateScatteringFunction(
                self._parser.read_array(self._metadata["data_offset"], shape),
                None,
                self.get_k(),
                self.get_tau(),
                self.get_bin_edges(),
            )

    def get_k(self) -> np.ndarray:
        """Read k array from file.

        Returns
        -------
        numpy.ndarray
            The k array.
        """
        offset = self._metadata["k_offset"]
        Nk = self._metadata["Nk"]

        return self._parser.read_array(offset, Nk)

    def get_tau(self) -> np.ndarray:
        """Read tau array from file.

        Returns
        -------
        numpy.ndarray
            The tau array.
        """
        offset = self._metadata["tau_offset"]
        Nt = self._metadata["Nt"]

        return self._parser.read_array(offset, Nt)

    def get_bin_edges(self) -> np.ndarray:
        """Read bin edges array from file.

        Returns
        -------
        numpy.ndarray
            The bin edges array.
        """
        offset = self._metadata["bin_edges_offset"]
        Nk = self._metadata["Nk"]

        return self._parser.read_array(offset, Nk)

    def get_k_slice(self, k_index: int) -> np.ndarray:
        """Read a slice at k from data.

        Parameters
        ----------
        k_index : int
            The k index

        Returns
        -------
        numpy.ndarray
            The data at k vs tau.

        Raises
        ------
        IndexError
            If k_index is out of range.
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

    def get_k_slice_err(self, k_index: int) -> Optional[np.ndarray]:
        """Read a slice of uncertainty at k from data.

        Parameters
        ----------
        k_index : int
            The k index

        Returns
        -------
        numpy.ndarray
            The uncertainty of data at k vs tau.

        Raises
        ------
        IndexError
            If k_index is out of range.
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


class ISFParser(Parser):
    """Intermediate scattering function file parser class.

    Inherits from `Parser`.
    """

    def __init__(self, fh: BinaryIO):
        super().__init__(fh)
        # check file identifier
        file_id = self._read_id()
        if file_id != 43:
            err_msg = (
                f"File identifier {file_id} not compatible with"
                " intermediate scattering function file (43)."
                " Input file might be wrong or corrupted."
            )
            raise RuntimeError(err_msg)

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


def melt(
    isf1: IntermediateScatteringFunction, isf2: IntermediateScatteringFunction
) -> IntermediateScatteringFunction:
    """Melt two intermediate scattering functions into one object.

    Parameters
    ----------
    isf1 : IntermediateScatteringFunction
        One IntermediateScatteringFunction object.
    isf2 : IntermediateScatteringFunction
        Another IntermediateScatteringFunction object.

    Returns
    -------
    IntermediateScatteringFunction
        The two IntermediateScatteringFunction objects, merged into a new one.
    """

    # assign fast and slow acquisition
    if isf1.tau[0] < isf2.tau[0]:
        fast, slow = isf1, isf2
    else:
        fast, slow = isf2, isf1

    # initialize tau and data
    Nt = min(10, np.sum(slow.tau < fast.tau[-1]))
    # keep data of `fast` up to (Nt/2)-th tau of `slow`
    idx = np.argmin(np.abs(fast.tau - slow.tau[Nt // 2])) + 1
    tau = np.append(fast.tau[:idx], slow.tau[Nt // 2 + 1 :]).astype(DTYPE)
    data = np.zeros((len(fast.k), len(tau)), dtype=DTYPE)
    if fast._err is None or slow._err is None:
        err = None
    else:
        err = np.zeros((len(fast.k), len(tau)), dtype=DTYPE)

    t = np.log(slow.tau[:Nt])

    # loop through k values
    for i in range(len(fast.k)):
        # check for nan
        if np.any(np.isnan([fast.data[i, 0], slow.data[i, 0]])):
            data[i] = np.full(len(tau), np.nan, dtype=DTYPE)
        else:
            # find multiplicative factor via least squares minimization
            # interpolate in loglog scale (smoother curve)
            f = interp1d(x=np.log(fast.tau), y=np.log(fast.data[i]), kind="cubic")
            sum_xiyi = np.sum(np.exp(f(t)) * slow.data[i, :Nt])
            sum_xi2 = np.sum(np.exp(f(t)) ** 2)
            alpha = sum_xiyi / sum_xi2

            # scale fast on slow
            data[i] = np.append(
                fast.data[i, :idx] * alpha, slow.data[i, Nt // 2 + 1 :]
            ).astype(DTYPE)
            if err is not None:
                err[i] = np.append(
                    fast.err[i, :idx] * alpha, slow.err[i, Nt // 2 + 1 :]  # type: ignore
                ).astype(DTYPE)

    # ensure DTYPE for all IntermediateScatteringFunction args
    k = fast.k.astype(DTYPE)
    bin_edges = fast.bin_edges.astype(DTYPE)

    return IntermediateScatteringFunction(data, err, k, tau, bin_edges)


def mergesort(
    isf1: IntermediateScatteringFunction, isf2: IntermediateScatteringFunction
) -> IntermediateScatteringFunction:
    """Merge the values of two intermediate scattering functions.
    Values will then be sorted based on tau.

    Parameters
    ----------
    isf1 : IntermediateScatteringFunction
        One IntermediateScatteringFunction object.
    isf2 : IntermediateScatteringFunction
        Another IntermediateScatteringFunction object.

    Returns
    -------
    IntermediateScatteringFunction
        The two IntermediateScatteringFunction objects are fused into a new one.
    """
    tau = np.append(isf1.tau, isf2.tau).astype(DTYPE)
    sortidx = np.argsort(tau)

    # create new data
    dim_k, dim_tau = isf1.shape
    data = np.zeros(shape=(dim_k, len(tau)), dtype=DTYPE)
    if isf1._err is None or isf2._err is None:
        err = None
    else:
        err = np.zeros(shape=(dim_k, len(tau)), dtype=DTYPE)

    # populate data
    data = np.append(isf1.data, isf2.data, axis=1)[:, sortidx].astype(DTYPE)
    if err is not None:
        err = np.append(isf1.err, isf2.err, axis=1)[:, sortidx].astype(  # type: ignore
            DTYPE
        )

    # ensure DTYPE for all IntermediateScatteringFunction args
    k = isf1.k.astype(DTYPE)
    bin_edges = isf1.bin_edges.astype(DTYPE)

    return IntermediateScatteringFunction(data, err, k, tau[sortidx], bin_edges)


def azavg2isf_estimate(
    az_avg: AzimuthalAverage,
    noise_est: str = "polyfit",
    plateau_est: str = "var",
    noise: Optional[np.ndarray] = None,
    noise_err: Optional[np.ndarray] = None,
    plateau: Optional[np.ndarray] = None,
    plateau_err: Optional[np.ndarray] = None,
    **kwargs,
) -> IntermediateScatteringFunction:
    """Convert AzimuthalAverage to IntermediateScatteringFunction

    Parameters
    ----------
    az_avg : AzimuthalAverage
        AzimuthalAverage object
    noise_est : str, optional
        Noise factor estimate mode, by default 'polyfit'. Accepted values are the ones
        supported by the estimate_camera_noise function of fastddm.noise_est module plus
        'custom'. In the latter case, noise input argument is required. Additional keyword
        arguments are used in the estimate_camera_noise function.
    plateau_est : str, optional
        Plateau estimate mode, by default 'var'. Accepted values are 'var', 'power_spec',
        or 'custom'. When 'var' ('power_spec') mode is selected, the plateau is estimated
        as twice the az_avg.var (az_avg.power_spec). When 'custom' is selected, the plateau
        input argument is required.
    noise : np.ndarray, optional
        Custom noise array, by default None. Required if noise_est is 'custom'
    noise_err : np.ndarray, optional
        Custom noise array uncertainty, by default None. Used if noise_est is 'custom'.
        If None and noise_est is 'custom', the noise uncertainty is assumed equal to the
        noise input array.
    plateau : np.ndarray, optional
        Custom plateau array, by default None. Required if plateau_est is 'custom'
    plateau_err : np.ndarray, optional
        Custom plateau array uncertainty, by default None. Used if plateau_est is 'custom'.
        If None and plateau_est is 'custom', the plateau uncertainty is assumed equal to the
        plateau input array.

    Returns
    -------
    IntermediateScatteringFunction
        IntermediateScatteringFunction

    Raises
    ------
    RuntimeError
        If the dimension of the input arrays are not compatible with the azimuthal average or
        if any of the estimate mode is not supported.
    """
    # get number of k values
    dim_k, dim_t = az_avg.shape

    # estimate noise (B)
    if noise_est == "custom":
        # sanity check on size of noise
        if noise is not None and len(noise) != dim_k:
            err_msg = (
                "Custom noise array dimension not compatible"
                " with given azimuthal average.\n"
                f"Size of noise should be {dim_k}."
            )
            raise RuntimeError(err_msg)

        if noise_err is None:
            noise_err = noise
        # sanity check on size of noise_err
        if noise_err is not None and len(noise_err) != dim_k:
            err_msg = (
                "Custom noise_err array dimension not compatible"
                " with given azimuthal average.\n"
                f"Size of noise_err should be {dim_k}."
            )
            raise RuntimeError(err_msg)
    else:
        noise, noise_err = estimate_camera_noise(az_avg, mode=noise_est, **kwargs)  # type: ignore
    # enforce dtype
    noise = noise.astype(DTYPE)  # type: ignore
    noise_err = noise_err.astype(DTYPE)  # type: ignore

    # estimate plateau (A+B)
    if plateau_est == "custom":
        # sanity check on size of plateau
        if plateau is not None and len(plateau) != dim_k:
            err_msg = (
                "Custom plateau array dimension not compatible"
                " with given azimuthal average.\n"
                f"Size of plateau should be {dim_k}."
            )
            raise RuntimeError(err_msg)

        if plateau_err is None:
            plateau_err = plateau
        # sanity check on size of plateau_err
        if plateau_err is not None and len(plateau_err) != dim_k:
            err_msg = (
                "Custom plateau_err array dimension not compatible"
                " with given azimuthal average.\n"
                f"Size of plateau_err should be {dim_k}."
            )
            raise RuntimeError(err_msg)
    elif plateau_est == "var":
        plateau = 2 * az_avg.var
        if az_avg.var_err is None:
            plateau_err = 2 * az_avg.var
        else:
            plateau_err = 2 * az_avg.var_err
    elif plateau_est == "power_spec":
        plateau = 2 * az_avg.power_spec
        if az_avg.power_spec_err is None:
            plateau_err = 2 * az_avg.power_spec
        else:
            plateau_err = 2 * az_avg.power_spec_err
    else:
        plateau_est_modes = ["custom", "power_spec", "var"]
        err_msg = (
            f"Unsupported plateau_est mode {plateau_est}."
            f" Possible values are {plateau_est_modes}"
        )
        raise RuntimeError(err_msg)
    # enforce dtype
    plateau = plateau.astype(DTYPE)  # type: ignore
    plateau_err = plateau_err.astype(DTYPE)  # type: ignore

    # convert structure function to intermediate scattering function
    data = np.zeros((dim_k, dim_t), dtype=DTYPE)
    err = np.zeros((dim_k, dim_t), dtype=DTYPE)
    for idx_k in range(dim_k):
        ApB = plateau[idx_k]  # A+B
        B = noise[idx_k]  # B
        A = ApB - B  # A

        sigma_ApB = plateau_err[idx_k]
        sigma_B = noise_err[idx_k]

        y = az_avg.data[idx_k].astype(DTYPE)
        if az_avg.err is None:
            sigma_y = y
        else:
            sigma_y = az_avg.err[idx_k].astype(DTYPE)
        # intermediate scattering function
        data[idx_k] = 1 - (y - B) / A
        # uncertainty
        sigma2 = ((y - B) / A**2 * sigma_ApB) ** 2
        sigma2 += ((ApB - y) / A**2 * sigma_B) ** 2
        sigma2 += (sigma_y / A) ** 2
        err[idx_k] = np.sqrt(sigma2)

    k = az_avg.k.astype(DTYPE)
    tau = az_avg.tau.astype(DTYPE)
    bin_edges = az_avg.bin_edges.astype(DTYPE)

    return IntermediateScatteringFunction(data, err, k, tau, bin_edges)
