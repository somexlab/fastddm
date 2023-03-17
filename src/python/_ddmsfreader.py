# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""Functions to write binary fastddm image structure function file.
"""

import struct
from os import path
from typing import BinaryIO, Tuple, Optional
from .imagestructurefunction import ImageStructureFunction
import numpy as np


format_byte_len = {
    'B' : 1,    # unsigned char
    'H' : 2,    # unsigned short
    'I' : 4,    # unsigned int
    'Q' : 8,    # unsigned long long
    'f' : 4,    # float
    'd' : 8,    # double
}

format2numpy = {
    'd' : np.float64,
    'f' : np.float32,
}


class DdmSFReader:
    """Wrapper for the fastddm image structure function
    binary file parser. Use this class to process your .sf.ddm files.
    """

    def __init__(self, file : str):
        if not file.endswith('.sf.ddm'):
            raise ValueError("Invalid file name extension. Expected .sf.ddm extension.")
        self.filename = file
        self._fh = open(file, "rb")

        self._parser = SFParser(self._fh)
        self.metadata = self._parser.metadata
        self._dtype = self._parser.get_dtype_from_metadata()

    def __enter__(self):
        # make file connection and return it
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the file is closed
        self.close()

    def close(self) -> None:
        """Close the file handle
        """
        if self._fh is not None:
            self._fh.close()

    def get_slice(self, index : int) -> np.ndarray:
        """Returns the image structure function slice at the selected index.

        Parameters
        ----------
        index : int
            Index.

        Returns
        -------
        np.ndarray
            The image structure function slice.
        """

        if index in range(self.metadata['Nt']):
            return self._parser._read_frame(index)
        else:
            return None

    def get_power_spec(self) -> np.ndarray:
        """Returns the power spectrum.

        Returns
        -------
        np.ndarray
            The power spectrum.
        """
        return self._parser._read_frame(self.metadata['Nt'])

    def get_var(self) -> np.ndarray:
        """Returns the variance.

        Returns
        -------
        np.ndarray
            The variance.
        """
        return self._parser._read_frame(self.metadata['Nt'] + 1)

    def get_kx(self) -> np.ndarray:
        """Creates a numpy array with kx values.

        Returns
        -------
        np.ndarray
            The kx array.
        """
        return self._parser._read_array(self.metadata['kx_offset'], self.metadata['Nx'])

    def get_ky(self) -> np.ndarray:
        """Creates a numpy array with ky values.

        Returns
        -------
        np.ndarray
            The ky array.
        """
        return self._parser._read_array(self.metadata['ky_offset'], self.metadata['Ny'])

    def get_tau(self) -> np.ndarray:
        """Creates a numpy array with tau values.

        Returns
        -------
        np.ndarray
            The tau array.
        """
        return self._parser._read_array(self.metadata['tau_offset'], self.metadata['Nt'])
    
    def get_image_structure_function(self) -> ImageStructureFunction:
        return ImageStructureFunction(
            self._parser._read_data(),
            self.get_kx(),
            self.get_ky(),
            self.get_tau(),
            self.metadata['pixel_size'],
            self.metadata['delta_t']
            )


class SFParser:
    """Parse fastddm image structure function file.
    """

    supported_file_versions = {(0, 1) : True}

    def __init__(self, fh):
        self._fh = fh
        self.metadata = {}

        # parse metadata
        self._read_byteorder_from_file()
        self._parse_metadata()

        # check if the file version is supported
        self.supported = self._check_version_supported()

    def _check_version_supported(self) -> bool:
        """Checks if the fastddm image structure function file version
        is supported by this reader.

        Returns
        -------
        bool
            True if supported.
        """
        major_version, minor_version = self.get_version()
        supported = self.supported_file_versions.get((major_version, minor_version)) or self.supported_file_versions.get((major_version, None))

        if not supported:
            print("Warning: No parser is available for your current .sf.ddm version " +
                  f"{major_version}.{minor_version}. " +
                  "This might lead to unexpected behavior.")

        return supported

    def _read_value(self, pos : int, format : chr, whence : Optional[int] = 0):
        self._fh.seek(pos, whence)
        Nbytes = format_byte_len[format]
        data = self._fh.read(Nbytes)
        if self.metadata['byteorder'] == 'big':
            return struct.unpack(f'>{format}', data)[0]
        else:
            return struct.unpack(f'<{format}', data)[0]

    def _read_array(self, pos : int, length : int) -> np.ndarray:
        # initialize array
        out = np.empty(length, dtype=self.get_dtype_from_metadata())

        # read values into array
        self._fh.seek(pos)
        Nbytes = format_byte_len[self.metadata['dtype']]
        if self.metadata['byteorder'] == 'big':
            fmt = f'>{self.metadata["dtype"]}'
        else:
            fmt = f'<{self.metadata["dtype"]}'
        for i in range(length):
            out[i], = struct.unpack(fmt, self._fh.read(Nbytes))

        return out


    def _read_frame(self, index : int) -> np.ndarray:
        Ny = self.metadata['Ny']
        Nx = self.metadata['Nx']
        # compute offset
        offset = index * Ny * Nx
        offset *= format_byte_len[self.metadata['dtype']]
        offset += self.metadata['data_offset']

        # initialize array
        out = np.empty(Nx*Ny, dtype=self.get_dtype_from_metadata())

        # read values into array
        self._fh.seek(offset)
        Nbytes = format_byte_len[self.metadata['dtype']]
        if self.metadata['byteorder'] == 'big':
            fmt = f'>{self.metadata["dtype"]}'
        else:
            fmt = f'<{self.metadata["dtype"]}'
        for i in range(Nx * Ny):
            out[i], = struct.unpack(fmt, self._fh.read(Nbytes))

        return out.reshape((Ny, Nx))

    def _read_data(self) -> np.ndarray:
        Nt = self.metadata['Nt'] + self.metadata['Nextra']
        Ny = self.metadata['Ny']
        Nx = self.metadata['Nx']

        offset = self.metadata['data_offset']

        # initialize array
        out = np.empty(Nt * Ny * Nx, dtype=self.get_dtype_from_metadata())

        # read values into array
        self._fh.seek(offset)
        Nbytes = format_byte_len[self.metadata['dtype']]
        if self.metadata['byteorder'] == 'big':
            fmt = f'>{self.metadata["dtype"]}'
        else:
            fmt = f'<{self.metadata["dtype"]}'
        for i in range(Nt * Ny * Nx):
            out[i], = struct.unpack(fmt, self._fh.read(Nbytes))

        return out.reshape((Nt, Ny, Nx))


    def get_version(self) -> Tuple[int, int]:
        """Determines version of the .sf.ddm file

        Returns
        -------
        Tuple[int, int]
            Major and minor version.
        """
        # the version starts after 4 bytes
        # the next two bytes contain the major and minor version packed as 2 unsigned char
        major_version = self._read_value(4,'B')
        minor_version = self._read_value(4 + format_byte_len['B'],'B')

        return major_version, minor_version
    
    def _read_byteorder_from_file(self) -> None:
        self._fh.seek(0)

        # endianness is at bytes 0-1, encoded as utf-8
        # 'II' : little-endian
        # 'MM' : big-endian
        endianness = self._fh.read(2).decode("utf-8")
        if endianness == 'II':
            self.metadata['byteorder'] = 'little'
        elif endianness == 'MM':
            self.metadata['byteorder'] = 'big'
        else:
            raise ValueError(f"Unsupported endianness {endianness} identifier.")

    def _parse_metadata(self) -> None:
        """Reads metadata and creates dictionary.
        """
        self._fh.seek(2)
        
        # length of image structure function data array
        # is at bytes 6-9, packed as unsigned int (I)
        self.metadata['Nt'] = self._read_value(6, 'I')

        # height of image structure function data array
        # is at bytes 10-13, packed as unsigned int (I)
        self.metadata['Ny'] = self._read_value(10, 'I')

        # width of image structure function data array
        # is at bytes 14-17, packed as unsigned int (I)
        self.metadata['Nx'] = self._read_value(14, 'I')

        # number of extra slices in image structure function data array
        # (i.e., variance and power spectrum)
        # is at bytes 18-21, packed as unsigned int (I)
        self.metadata['Nextra'] = self._read_value(18, 'I')

        # dtype identifier
        # is at byte 22, packed as char
        self.metadata['dtype'] = self._fh.read(1).decode("utf-8")

        # data byte offset
        self.metadata['data_offset'] = self._read_value(-format_byte_len['Q'], 'Q', 2)

        # kx byte offset
        self.metadata['kx_offset'] = self._read_value(-2 * format_byte_len['Q'], 'Q', 2)

        # ky byte offset
        self.metadata['ky_offset'] = self._read_value(-3 * format_byte_len['Q'], 'Q', 2)

        # tau byte offset
        self.metadata['tau_offset'] = self._read_value(-4 * format_byte_len['Q'], 'Q', 2)

        # pixel size byte offset
        pixel_size_offset = self._read_value(-5 * format_byte_len['Q'], 'Q', 2)

        # delta_t byte offset
        delta_t_offset = self._read_value(-6 * format_byte_len['Q'], 'Q', 2)

        # extra byte offset
        self.metadata['extra_offset'] = self._read_value(-7 * format_byte_len['Q'], 'Q', 2)

        # pixel_size
        self.metadata['pixel_size'] = self._read_value(pixel_size_offset, self.metadata['dtype'], 0)

        # delta_t
        self.metadata['delta_t'] = self._read_value(delta_t_offset, self.metadata['dtype'], 0)


    def get_dtype_from_metadata(self) -> np.dtype:
        return format2numpy[self.metadata['dtype']]
