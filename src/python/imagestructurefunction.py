# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

r"""This module contains the image structure function data class.

The :py:class:`ImageStructureFunction` object is used to store and retrieve information
about the image structure function computed in DDM.

The :py:class:`ImageStructureFunction.data` contains the image structure function
values in :math:`(\Delta t, k_y, k_x)` order. For instance, the image
structure function at the 10th delay computed can be accessed via

.. code-block:: python

    dqt.data[9]

.. note::
   Remember that Python uses zero-based indexing.

The :py:class:`ImageStructureFunction` can then be saved into a binary file by using
:py:meth:`ImageStructureFunction.save` (it will have a ``.sf.ddm`` extension)
and later retrieved from the memory using
:py:meth:`SFReader.load`, which you can call directly from ``fastddm`` as

.. code-block:: python

    # load image structure function
    dqt = fastddm.load('path/to/my_dqt_file.sf.ddm')

In order to avoid reading and loading the entire file, we also provide
a fast reader through the :py:class:`SFReader`, which can be used to
access directly from the disk the relevant data, for example:

.. code-block:: python

    from fastddm.imagestructurefunction import SFReader

    # open file
    r = SFReader('path/to/my_dqt_file.sf.ddm')

    # access quantities in full plane representation
    # access kx array
    kx = r.get_kx()
    # access 10th computed delay
    frame = r.get_frame(index=9)

    # access the same quantities in half plane representation
    kx_half = r.get_kx(full=False)
    frame_half = r.get_frame(index=9, full=False)
"""

from typing import Sequence, Tuple, BinaryIO, Optional
from dataclasses import dataclass
import os
from sys import byteorder
import struct
import numpy as np

from ._io_common import (
    _save_as_tiff,
    calculate_format_size,
    npdtype2format,
    Writer,
    Reader,
    Parser,
)


@dataclass
class ImageStructureFunction:
    """Image structure function container class.

    Parameters
    ----------
    _data : numpy.ndarray
        The packed data (2D image structure function, power spectrum,
        and variance).
    _kx : numpy.ndarray
        The array of wavevector values over `x`.
    _ky : numpy.ndarray
        The array of wavevector values over `y`.
    _width : int
        The width of the full (symmetric) 2D image structure function.
    _height : int
        The height of the full (symmetric) 2D image structure function.
    _tau : numpy.ndarray
        The array of time delays.
    _pixel_size : float, optional
        The effective pixel size. Default is 1.
    _delta_t : float, optional
        The time delay between two consecutive frames. Default is 1.
    """

    _data: np.ndarray
    _kx: np.ndarray
    _ky: np.ndarray
    _width: int
    _height: int
    _tau: np.ndarray
    _pixel_size: float = 1.0
    _delta_t: float = 1.0

    @property
    def data(self) -> np.ndarray:
        """The 2D image structure function
        (in :math:`(\Delta t, k_y, k_x)` order).

        Returns
        -------
        numpy.ndarray
            The 2D image structure function.
        """
        return self._data[:-2]

    @property
    def kx(self) -> np.ndarray:
        """The array of wave vector values over `x`.

        Returns
        -------
        numpy.ndarray
            The array of ``kx``.
        """
        return self._kx

    @property
    def ky(self) -> np.ndarray:
        """The array of wave vector values over `y`.

        Returns
        -------
        numpy.ndarray
            The array of ``ky``.
        """
        return self._ky

    @property
    def width(self) -> int:
        """The width of the full (symmetric) 2D image structure function.

        Returns
        -------
        int
            The full width.
        """
        return self._width

    @property
    def height(self) -> int:
        """The height of the full (symmetric) 2D image structure function.

        Returns
        -------
        int
            The full height.
        """
        return self._height

    @property
    def tau(self) -> np.ndarray:
        """The array of time delays.

        Returns
        -------
        numpy.ndarray
            The array of ``tau``.
        """
        return self._tau

    @property
    def power_spec(self) -> np.ndarray:
        """The average 2D power spectrum of the input images.

        Returns
        -------
        numpy.ndarray
            The average 2D power spectrum of the input images.
        """
        return self._data[-2]

    @property
    def var(self) -> np.ndarray:
        """The variance (over time) of the Fourier transformed input images.

        Returns
        -------
        numpy.ndarray
            The variance (over time) of the Fourier transformed input images.
        """
        return self._data[-1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of image structure function data.

        Returns
        -------
        Tuple[int, int, int]
            The shape of the data.
        """
        return self.data.shape

    @property
    def pixel_size(self) -> float:
        """The effective pixel size.

        Returns
        -------
        float
            Pixel size.
        """
        return self._pixel_size

    @property
    def delta_t(self) -> float:
        """The time delay between to consecutive frames.

        Returns
        -------
        float
            Time delay.
        """
        return self._delta_t

    @pixel_size.setter
    def pixel_size(self, pixel_size: float) -> None:
        """Set the image effective pixel size.

        This will propagate also on the values of ``kx`` and ``ky``.

        Parameters
        ----------
        pixel_size : float
            The effective pixel size.
        """
        self._kx *= self._pixel_size / pixel_size
        self._ky *= self._pixel_size / pixel_size
        self._pixel_size = pixel_size

    @delta_t.setter
    def delta_t(self, delta_t: float) -> None:
        """Set the time delay between two consecutive frames.

        This will propagate also on the values of ``tau``.

        Parameters
        ----------
        delta_t : float
            The time delay.
        """
        self._tau *= delta_t / self._delta_t
        self._delta_t = delta_t

    def __len__(self):
        """The length of the image structure function data.
        It coincides with the number of lags.

        Returns
        -------
        int
            The length of data.
        """
        return len(self.data)

    def set_frame_rate(self, frame_rate: float) -> None:
        """Set the acquisition frame rate.

        This will propagate also on the values of ``tau``.

        Parameters
        ----------
        frame_rate : float
            The acquisition frame rate.
        """
        self.delta_t = 1 / frame_rate

    def save(self, fname: str = "analysis_blob") -> None:
        """Save ``ImageStructureFunction`` to binary file.

        Parameters
        ----------
        fname : str, optional
            The full file name, by default "analysis_blob".
        """
        # check name
        dir, name = os.path.split(fname)
        name = name if name.endswith(".sf.ddm") else f"{name}.sf.ddm"

        # save to file
        with SFWriter(file=os.path.join(dir, name)) as f:
            f.write_obj(self)

    def save_as_tiff(self, seq: Sequence[int], fnames: Sequence[str]) -> None:
        """Save ``ImageStructureFunction`` frames as images.

        Parameters
        ----------
        seq : Optional[Sequence[int]]
            List of indices to export.
        fnames : Optional[Sequence[str]], optional
            List of file names.

        Raises
        ------
        RuntimeError
            If number of elements in ``fnames`` and ``seq`` are different.
        """
        if len(fnames) != len(seq):
            raise RuntimeError("Number of elements in fnames differs from one in seq.")

        for i, f in zip(seq, fnames):
            _save_as_tiff(data=self.full_slice(i), labels=[f])

    def full_shape(self) -> Tuple[int, int, int]:
        """The shape of the full (symmetric) 2D image structure function.

        Returns
        -------
        Tuple[int, int, int]
            The full shape.
        """
        dim_t, dim_y = self.shape[:-1]
        dim_x = self._width
        return dim_t, dim_y, dim_x

    def full_slice(self, idx: int) -> np.ndarray:
        """Get the full (symmetric) 2D image structure function at index ``idx``.

        Parameters
        ----------
        idx : int
            The slice index.

        Returns
        -------
        numpy.ndarray
            The full 2D image structure function slice.

        Raises
        ------
        IndexError
            If ``idx`` is out of bounds.
        """
        if idx >= 0 and idx < len(self.data):
            shape = (self.height, self.width)
            return _reconstruct_full_spectrum(self._data[idx], shape)
        else:
            raise IndexError(
                f"Index out of range. Choose an index between 0 and {len(self.data)}."
            )

    def full_power_spec(self) -> np.ndarray:
        """Get the full (symmetric) average 2D power spectrum of the input images.

        Returns
        -------
        numpy.ndarray
            The full 2D power spectrum.
        """
        shape = (self.height, self.width)
        return _reconstruct_full_spectrum(self._data[-2], shape)

    def full_var(self) -> np.ndarray:
        """Get the full (symmetric) 2D variance (over time) of the Fourier
        transformed images.

        Returns
        -------
        numpy.ndarray
            The full 2D variance.
        """
        shape = (self.height, self.width)
        return _reconstruct_full_spectrum(self._data[-1], shape)

    def full_kx(self) -> np.ndarray:
        """Get the full array of wavevector values over `x`.

        Returns
        -------
        numpy.ndarray
            The full ``kx`` array.
        """
        # initialize output and set dim
        full_kx = np.zeros_like(self.kx, shape=(self.width))
        dim_x = len(self.kx)
        # copy first part
        full_kx[:dim_x] = self.kx
        # copy other half
        if self.width % 2 == 0:
            full_kx[dim_x:] = -np.flip(self.kx[1:-1])
        else:
            full_kx[dim_x:] = -np.flip(self.kx[1:])

        return np.fft.fftshift(full_kx)

    def full_ky(self) -> np.ndarray:
        """Get the full array of wavevector values over `y`.

        Returns
        -------
        numpy.ndarray
            The full ``ky`` array.
        """
        return self.ky


class SFWriter(Writer):
    """Image structure function writer class. Inherits from ``Writer``.

    It adds the unique method ``write_obj``.

    Defines the functions to write :py:class:`ImageStructureFunction` object to binary file.

    The structure of the binary file is the following:

    Header:

    - bytes 0-1: endianness, string, utf-8 encoding [``"LL"`` = 'little', ``"BB"`` = 'big']
    - bytes 2-3: file identifier, 16-bit integer, unsigned short [``73`` for image structure function]
    - bytes 4-5: file version, pair of 8-bit integers as (major_version, minor_version), unsigned char
    - byte 6: dtype, string, utf-8 encoding [``"d"`` = float64, ``"f"`` = float32]
    - bytes 7-14: data length, 64-bit integer, unsigned long long
    - bytes 15-22: data height, 64-bit integer, unsigned long long
    - bytes 23-30: data width, 64-bit integer, unsigned long long
    - bytes 31-38: extra slices, 64-bit integer, unsigned long long
    - bytes 39-46: full width, 64-bit integer, unsigned long long
    - bytes 47-54: full height, 64-bit integer, unsigned long long

    The data is stored in 'C' order and ``dtype`` format as follows:

    - from byte ``data_offset``: ``_data``
    - from byte ``extra_offset``: extra data
    - from byte ``kx_offset``: ``kx`` array
    - from byte ``ky_offset``: ``ky`` array
    - from byte ``tau_offset``: ``tau`` array
    - from byte ``pixel_size_offset``: ``pixel_size`` value
    - from byte ``delta_t_offset``: ``delta_t`` value

    From the end of the file,
    the byte offsets are stored as unsigned long long 64-bit integers in this order:

    - ``data_offset``
    - ``kx_offset``
    - ``ky_offset``
    - ``tau_offset``
    - ``pixel_size_offset``
    - ``delta_t_offset``
    - ``extra_offset``
    """

    def write_obj(self, obj: ImageStructureFunction) -> None:
        """Write :py:class:`ImageStructureFunction` object to binary file.

        Parameters
        ----------
        obj : ImageStructureFunction
            ``ImageStructureFunction`` object.
        """
        # get data dtype
        dtype = npdtype2format(obj.data.dtype.name)
        # get data shape
        Nt, Ny, Nx = obj.shape
        # assign Nextra = 2
        # we have power_spectrum + variance
        Nextra = len(obj._data) - len(obj)
        # get full width and height
        width = obj.width
        height = obj.height

        # write file header
        self._write_header(Nt, Ny, Nx, Nextra, width, height, dtype)

        # write data
        self._write_data(obj)

    def _write_header(
        self,
        Nt: int,
        Ny: int,
        Nx: int,
        Nextra: int,
        width: int,
        height: int,
        dtype: str,
    ) -> None:
        """Write image structure function file header.

        In version 0.3, the header is structured as follows:
        * bytes 0-1: endianness (`LL` = 'little'; `BB` = 'big'), 'utf-8' encoding
        * bytes 2-3: file identifier (73), `H` (unsigned short)
        * bytes 4-5: file version as (major_version, minor_version), `BB` (unsigned char)
        * byte 6: dtype (`d` = float64; `f` = float32), 'utf-8' encoding
        * bytes 7-14: data length, `Q` (unsigned long long)
        * bytes 15-22: data height, `Q` (unsigned long long)
        * bytes 23-30: data width, `Q` (unsigned long long)
        * bytes 31-38: extra slices, `Q` (unsigned long long)
        * bytes 39-46: full width, `Q` (unsigned long long)
        * bytes 47-54: full height, `Q` (unsigned long long)

        Parameters
        ----------
        Nt : int
            Length.
        Ny : int
            Height.
        Nx : int
            Width
        Nextra : int
            Number of extra slices.
        width : int
            Full width.
        height : int
            Full height.
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
        self._fh.write(struct.pack("H", 73))
        curr_byte_len += calculate_format_size("H")

        # file version
        self._fh.write(struct.pack("BB", *(self.version)))
        curr_byte_len += 2 * calculate_format_size("B")

        # dtype
        self._fh.write(dtype.encode("utf-8"))
        curr_byte_len += 1

        # length
        self._fh.write(struct.pack("Q", Nt))
        curr_byte_len += calculate_format_size("Q")

        # height
        self._fh.write(struct.pack("Q", Ny))
        curr_byte_len += calculate_format_size("Q")

        # width
        self._fh.write(struct.pack("Q", Nx))
        curr_byte_len += calculate_format_size("Q")

        # extra slices
        self._fh.write(struct.pack("Q", Nextra))
        curr_byte_len += calculate_format_size("Q")

        # full width
        self._fh.write(struct.pack("Q", width))
        curr_byte_len += calculate_format_size("Q")

        # full height
        self._fh.write(struct.pack("Q", height))
        curr_byte_len += calculate_format_size("Q")

        # add empty bytes up to HEAD_BYTE_LEN for future use (if needed)
        self._fh.write(bytearray(self.head_byte_len - curr_byte_len))

    def _write_data(self, obj: ImageStructureFunction) -> None:
        """Write image structure function data.

        In version 0.1, the data is stored in 'C' order and `dtype` format as follows:
        * from `data_offset`: _data
        * from `extra_offset`: extra data
        * from `kx_offset`: `kx` array
        * from `ky_offset`: `ky` array
        * from `tau_offset`: `tau` array
        * from `pixel_size_offset`: `pixel_size` value
        * from `delta_t_offset`: `delta_t` value

        From the end of the file,
        the byte offsets are stored in `Q` (unsigned long long) format in this order:
        * `data_offset`
        * `kx_offset`
        * `ky_offset`
        * `tau_offset`
        * `pixel_size_offset`
        * `delta_t_offset`
        * `extra_offset`

        Parameters
        ----------
        obj : ImageStructureFunction
            The image structure function object.
        """
        # get data format
        fmt = npdtype2format(obj.data.dtype.name)

        # get data shape
        Nt, Ny, Nx = obj.shape
        Nextra = len(obj._data) - len(obj)

        # write _data
        obj._data.tofile(self._fh)
        data_offset = self.head_byte_len
        extra_offset = data_offset + Nt * Ny * Nx * calculate_format_size(fmt)

        # write kx, ky, and tau
        obj.kx.tofile(self._fh)
        obj.ky.tofile(self._fh)
        obj.tau.tofile(self._fh)
        kx_offset = extra_offset + Nextra * Ny * Nx * calculate_format_size(fmt)
        ky_offset = kx_offset + Nx * calculate_format_size(fmt)
        tau_offset = ky_offset + Ny * calculate_format_size(fmt)

        # write pixel_size and delta_t
        self._fh.write(struct.pack(fmt, obj.pixel_size))
        self._fh.write(struct.pack(fmt, obj.delta_t))
        pixel_size_offset = tau_offset + Nt * calculate_format_size(fmt)
        delta_t_offset = pixel_size_offset + calculate_format_size(fmt)

        # write byte offsets
        self._fh.write(struct.pack("Q", extra_offset))
        self._fh.write(struct.pack("Q", delta_t_offset))
        self._fh.write(struct.pack("Q", pixel_size_offset))
        self._fh.write(struct.pack("Q", tau_offset))
        self._fh.write(struct.pack("Q", ky_offset))
        self._fh.write(struct.pack("Q", kx_offset))
        self._fh.write(struct.pack("Q", data_offset))


class SFReader(Reader):
    """FastDDM image structure function reader class.
    Inherits from ``Reader``.
    """

    def __init__(self, file: str):
        super().__init__(file)
        self._parser = SFParser(self._fh)
        self._metadata = self._parser.read_metadata()

    def load(self) -> ImageStructureFunction:
        """Load image structure function from file.

        Returns
        -------
        ImageStructureFunction
            ``ImageStructureFunction`` object.

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
        Nt = self._metadata["Nt"]
        Nextra = self._metadata["Nextra"]
        Ny = self._metadata["Ny"]
        Nx = self._metadata["Nx"]
        shape = (Nt + Nextra, Ny, Nx)

        return ImageStructureFunction(
            self._parser.read_array(self._metadata["data_offset"], shape),
            self.get_kx(False),
            self.get_ky(False),
            self._metadata["width"],
            self._metadata["height"],
            self.get_tau(),
            self._metadata["pixel_size"],
            self._metadata["delta_t"],
        )

    def get_kx(self, full: Optional[bool] = True) -> np.ndarray:
        """Read ``kx`` array from file.

        Parameters
        ----------
        full : Optional[bool]
            If True, return the full (symmetric) ``kx`` array. Default is True.

        Returns
        -------
        numpy.ndarray
            The ``kx`` array.
        """
        offset = self._metadata["kx_offset"]
        Nx = self._metadata["Nx"]

        kx = self._parser.read_array(offset, Nx)

        if full:
            out = np.zeros_like(kx, shape=self._metadata["width"])
            # copy first part
            out[:Nx] = kx
            # copy other half
            if self._metadata["width"] % 2 == 0:
                out[Nx:] = -np.flip(kx[1:-1])
            else:
                out[Nx:] = -np.flip(kx[1:])
            return np.fft.fftshift(out)
        else:
            return kx

    def get_ky(self, full: Optional[bool] = True) -> np.ndarray:
        """Read ``ky`` array from file.

        Parameters
        ----------
        full : Optional[bool]
            If True, return the full (symmetric) ``ky`` array. Default is True.
            This flag has in fact no effect on the output since ``ky`` is already
            full.

        Returns
        -------
        numpy.ndarray
            The ``ky`` array.
        """
        offset = self._metadata["ky_offset"]
        Ny = self._metadata["Ny"]

        return self._parser.read_array(offset, Ny)

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

    def get_frame(self, index: int, full: Optional[bool] = True) -> np.ndarray:
        """Read ``data`` slice array from file.

        Parameters
        ----------
        index : int
            The frame index.
        full : Optional[bool]
            If True, return the full (symmetric) 2D image structure function.
            Default is True.

        Returns
        -------
        numpy.ndarray
            The ``data`` slice array.

        Raises
        ------
        IndexError
            If ``index`` is out of range.
        """
        # check index is in range
        Nt = self._metadata["Nt"]
        if index < 0 or index >= Nt:
            raise IndexError(
                f"Index out of range. Choose an index between 0 and {Nt-1}."
            )

        Nx = self._metadata["Nx"]
        Ny = self._metadata["Ny"]
        offset = self._metadata["data_offset"]
        offset += index * Nx * Ny * calculate_format_size(self._parser.dtype)

        if full:
            shape = (self._metadata["height"], self._metadata["width"])
            return _reconstruct_full_spectrum(
                self._parser.read_array(offset, (Ny, Nx)), shape
            )
        else:
            return self._parser.read_array(offset, (Ny, Nx))

    def get_power_spec(self, full: Optional[bool] = True) -> np.ndarray:
        """Read power spectrum array from file.

        Parameters
        ----------
        full : Optional[bool]
            If True, return the full (symmetric) power spectrum.
            Default is True.

        Returns
        -------
        numpy.ndarray
            The power spectrum array.
        """
        offset = self._metadata["extra_offset"]
        Nx = self._metadata["Nx"]
        Ny = self._metadata["Ny"]

        if full:
            shape = (self._metadata["height"], self._metadata["width"])
            return _reconstruct_full_spectrum(
                self._parser.read_array(offset, (Ny, Nx)), shape
            )
        else:
            return self._parser.read_array(offset, (Ny, Nx))

    def get_var(self, full: Optional[bool] = True) -> np.ndarray:
        """Read variance array from file.

        Parameters
        ----------
        full : Optional[bool]
            If True, return the full (symmetric) variance.
            Default is True.

        Returns
        -------
        numpy.ndarray
            The variance array.
        """
        offset = self._metadata["extra_offset"]
        Nx = self._metadata["Nx"]
        Ny = self._metadata["Ny"]
        offset += Nx * Ny * calculate_format_size(self._parser.dtype)

        if full:
            shape = (self._metadata["height"], self._metadata["width"])
            return _reconstruct_full_spectrum(
                self._parser.read_array(offset, (Ny, Nx)), shape
            )
        else:
            return self._parser.read_array(offset, (Ny, Nx))


class SFParser(Parser):
    """Image structure function file parser class.
    Inherits from ``Parser``.
    """

    def __init__(self, fh: BinaryIO):
        super().__init__(fh)
        # check file identifier
        file_id = self._read_id()
        if file_id != 73:
            err_str = f"File identifier {file_id} not compatible with"
            err_str += " image structure function file (73)."
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

        # shape starts at byte 7
        # it comprises 4 values (Nt, Ny, Nx, Nextra), written as
        # unsigned long long ('Q')
        metadata["Nt"] = self.read_value(7, "Q")
        metadata["Ny"] = self.read_value(0, "Q", whence=1)
        metadata["Nx"] = self.read_value(0, "Q", whence=1)
        metadata["Nextra"] = self.read_value(0, "Q", whence=1)
        metadata["width"] = self.read_value(0, "Q", whence=1)
        metadata["height"] = self.read_value(0, "Q", whence=1)

        # byte offsets start from end of file, written as
        # unsigned long long ('Q')
        Q_size = calculate_format_size("Q")
        metadata["data_offset"] = self.read_value(-Q_size, "Q", 2)
        metadata["kx_offset"] = self.read_value(-2 * Q_size, "Q", 1)
        metadata["ky_offset"] = self.read_value(-2 * Q_size, "Q", 1)
        metadata["tau_offset"] = self.read_value(-2 * Q_size, "Q", 1)
        pixel_size_offset = self.read_value(-2 * Q_size, "Q", 1)
        delta_t_offset = self.read_value(-2 * Q_size, "Q", 1)
        metadata["extra_offset"] = self.read_value(-2 * Q_size, "Q", 1)

        # read pixel_size and delta_t
        metadata["pixel_size"] = self.read_value(pixel_size_offset, self.dtype)
        metadata["delta_t"] = self.read_value(delta_t_offset, self.dtype)

        return metadata


def _reconstruct_full_spectrum(
    halfplane: np.ndarray, shape: Tuple[int, int]
) -> np.ndarray:
    """Reconstruct the full (symmetric) 2D image structure function from the
    half plane representation.

    We keep half plane to save memory while computing. The result is the same as
    one would get by doing all calculations using scipy.fft.fft2 (or similar)
    and fftshift. The input is assumed to have the same format of the output of
    scipy.fft.rfft (or similar). The final `shape` must be given in order to
    correctly reconstruct the full representation.

    Parameters
    ----------
    halfplane : np.ndarray
        The half-plane array.
    shape : Tuple[int, ...]
        The shape of the full spectrum.

    Returns
    -------
    np.ndarray
        The full-plane 2D spectrum.
    """

    # setup of dtype and dimensions
    dtype = halfplane.dtype
    height, width = shape
    dim_y, dim_x = halfplane.shape

    # create full
    spectrum = np.zeros(shape, dtype=dtype)

    # copy half plane
    spectrum[:, :dim_x] = halfplane

    # other half is flipped
    # in particlar:
    # - if x is even, the flip is done on the columns from index 1 to index
    #       `dim_x`-1 (excl.), and from 1 to `dim_x` (excl.) otherwise
    # - if y is even, the flip is done on the rows from index 1 on, and from
    #       0 on otherwise. In the first case, the first row is copied in
    #       reversed order consistent with previous x-range condition

    # compute start and end indices
    x_s = 1
    x_e = dim_x - 1 if width % 2 == 0 else dim_x
    y_s = 1 if height % 2 == 0 else 0
    y_e = height

    spectrum[y_s:y_e, dim_x:] = np.flip(spectrum[y_s:y_e, x_s:x_e])

    if height % 2 == 0:
        spectrum[0, dim_x:] = np.flip(spectrum[0, x_s:x_e])

    spectrum = np.fft.fftshift(spectrum, axes=-1)

    return spectrum
