# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

"""Collection of common functions to write and read binary files."""

from typing import BinaryIO, Tuple, Optional, Any, Union, Sequence
import struct
from pathlib import Path
from os.path import dirname
import numpy as np
import tifffile


VERSION = (0, 3)
HEAD_BYTE_LEN = 64

def calculate_format_size(fmt : str) -> int:
    """Returns the size (in bytes) corresponding to the given format string.

    Parameters
    ----------
    fmt : str
        Format string.

    Returns
    -------
    int
        Size (in bytes) of format.
    """
    return struct.calcsize(fmt)

def npdtype2format(dtype_str : str) -> str:
    """Convert numpy dtype to format.

    Parameters
    ----------
    dtype_str : str
        The numpy dtype name.

    Returns
    -------
    str
        Format string.
    """
    if dtype_str == 'float64':
        return 'd'
    if dtype_str == 'float32':
        return 'f'
    raise NotImplementedError(f'Numpy dtype to binary format converter not implemented for type {dtype_str}.')


class Writer:
    """FastDDM generic writer class.

    Parameters
    ----------
    file : str
        Path string to the output file.

    Attributes
    ----------
    _fh : BinaryIO
        The handle to the binary file.
    version : Tuple[int, int]
        File version as (major_v, minor_v).
    head_byte_len : int
        Length (in bytes) of the header.

    Methods
    -------
    close() : None
        Close the file handle.
    """
    version = VERSION
    head_byte_len = HEAD_BYTE_LEN


    def __init__(self, file : str):
        """Writer constructor.

        It takes a file path string as an argument, creates the parent directory if
        it does not exist, and finally opens the file in binary write mode.

        Parameters
        ----------
        file : str
            Output file path string.
        """
        # ensure that storage folder exists
        dir_name = Path(dirname(file))
        dir_name.mkdir(parents=True, exist_ok=True)

        # open file in binary write mode with buffering disabled
        self._fh = open(file, 'wb', 0)


    def __enter__(self):
        """Context manager __enter__ method.
        """
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager __exit__ method.
        """
        self.close()


    def close(self) -> None:
        """Close the file handle.
        """
        if self._fh:
            self._fh.close()


class Reader:
    """FastDDM generic reader class.

    Parameters
    ----------
    file : str
        Path string to the input file.

    Attributes
    ----------
    _fh : BinaryIO
        The handle to the binary file.

    Methods
    -------
    close() : None
        Close the file handle.
    """

    def __init__(self, file : str):
        """Reader constructor.

        It takes a file path string as an argument and opens the file in binary read mode.

        Parameters
        ----------
        file : str
            Input file path string.
        """
        # open file in binary read mode
        self._fh = open(file, 'rb')


    def __enter__(self):
        """Context manager __enter__ method.
        """
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager __exit__ method.
        """
        self.close()


    def close(self) -> None:
        """Close the file handle.
        """
        if self._fh:
            self._fh.close()


class Parser:
    """FastDDM generic parser class.

    Parameters
    ----------
    fh : BinaryIO
        Handle to the input file.

    Attributes
    ----------
    _fh : BinaryIO
        The handle to the binary file.
    byteorder : str
        The input file byte order. Possible values are 'little' or 'big'.
    dtype : str
        The data dtype.
    supported : bool
        True if file version is supported by this parser.
    supported_file_versions : dict
        Dictionary of supported file versions.   

    Methods
    -------
    check_version_supported() : bool
        Returns True if the file version is supported by this parser.
    get_version() : Tuple[int, int]
        Returns the file version.
    read_value(offset, fmt, whence) : Any
        Reads a single value from the file.
    read_array(offset, shape) : np.ndarray
        Reads a ndarray from the file.
    """

    supported_file_versions = {(0, 1) : False, (0, 2) : False, (0, 3) : True}

    def __init__(self, fh : BinaryIO):
        self._fh = fh
        self.byteorder = self._read_byteorder()
        self.dtype = self._read_dtype()
        self.supported = self.check_version_supported()


    def check_version_supported(self) -> bool:
        """Check if the file version is supported by this parser.

        Returns
        -------
        bool
            True if supported.
        """
        major_version, minor_version = self.get_version()
        supported = self.supported_file_versions.get((major_version, minor_version)) or self.supported_file_versions.get((major_version, None))

        if not supported:
            print("Warning: No parser is available for your current file version " +
                  f"{major_version}.{minor_version}. " +
                  "This might lead to unexpected behavior.")

        return supported


    def _read_byteorder(self) -> str:
        """Read byte order flag from binary input file.

        Returns
        -------
        str
            The byte order flag.
            Possible output values are 'little' for little-endian
            or 'big' for big-endian byte order.

        Raises
        ------
        RuntimeError
            If byte order flag is not understood.
        """
        # endianness is at bytes 0-1, encoded as utf-8
        # 'LL' : little-endian
        # 'BB' : big-endian
        self._fh.seek(0)
        
        flag = self._fh.read(2).decode('utf-8')
        if flag == 'LL':
            return 'little'
        if flag == 'BB':
            return 'big'
        raise RuntimeError(f'Byteorder {flag} not recognized. Input file might be corrupted.')
    

    def _read_dtype(self) -> str:
        """Read the dtype format from the binary input file.

        Returns
        -------
        str
            The dtype format flag.
        """
        # dtype is at byte 6, encoded as utf-8
        self._fh.seek(6)
        return self._fh.read(1).decode('utf-8')


    def get_version(self) -> Tuple[int, int]:
        """Get version of the file.

        Returns
        -------
        Tuple[int, int]
            Major and minor version.
        """
        # version is always at bytes 4 and 5
        major_version = self.read_value(4, 'B')
        minor_version = self.read_value(5, 'B')
        return major_version, minor_version


    def read_value(self, offset : int, fmt : str, whence : Optional[int] = 0) -> Any:
        """Read a single value from the binary file.

        Parameters
        ----------
        offset : int
            Byte offset
        fmt : str
            Format.
        whence : Optional[int], optional
            Whence, by default 0 which means absolute file positioning.
            Other values are 1 which means seek relative to the current position
            and 2 means seek relative to the file's end. 

        Returns
        -------
        Any
            Read value.
        """
        self._fh.seek(offset, whence)
        data = self._fh.read(calculate_format_size(fmt))
        return struct.unpack(self._full_fmt(fmt), data)[0]


    def read_array(self, offset : int, shape : Union[int, Tuple[int, ...]]) -> np.ndarray:
        """Read a multidimensional array of values from the binary file.

        Parameters
        ----------
        offset : int
            Byte offset.
        shape : Union[int, Tuple[int, ...]]
            The shape of the array to be read.

        Returns
        -------
        np.ndarray
            The output array.
        """
        self._fh.seek(0)
        count = np.prod(shape)
        dtype = self._full_fmt(self.dtype)

        return np.fromfile(self._fh, dtype, count, offset=offset).reshape(shape)


    def _read_id(self) -> int:
        """Read file identifier from header.

        Returns
        -------
        int
            The file identifier.
        """
        # id is always at bytes 2-3, written as unsigned short ('H')
        return self.read_value(2, 'H')


    def _full_fmt(self, fmt : str) -> str:
        """Return full format with byte order flag

        Parameters
        ----------
        fmt : str
            The format.

        Returns
        -------
        str
            The full format.
        """
        if self.byteorder == 'little':
            return f'<{fmt}'
        if self.byteorder == 'big':
            return f'>{fmt}'


def _save_as_tiff(
    data : np.ndarray,
    labels : Sequence[str]
    ) -> None:
    """Save 3D numpy array as tiff image sequence.

    Parameters
    ----------
    data : np.ndarray
        The input array to be saved
    labels : Sequence[str]
        List of file names.
    """
    # check directory exists
    dir_name = Path(dirname(labels[0]))
    dir_name.mkdir(parents=True, exist_ok=True)

    for i, label in enumerate(labels):
        tifffile.imwrite(label, data[i].astype(np.float32))
