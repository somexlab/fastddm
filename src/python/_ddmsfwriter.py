# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""Functions to write binary fastddm image structure function file.
"""

import struct
from os import path, makedirs
from typing import BinaryIO
from sys import byteorder
from .imagestructurefunction import ImageStructureFunction

VERSION = (0, 1)
HEADER_BYTE_LEN = 64

format_byte_len = {
    'B' : 1,    # unsigned char
    'H' : 2,    # unsigned short
    'I' : 4,    # unsigned int
    'Q' : 8,    # unsigned long long
    'f' : 4,    # float
    'd' : 8,    # double
}

class DdmSFWriter:
    """
    FastDDM image structure function writer class.
    """

    version = VERSION

    def __init__(self, file : str):
        # create directory if it does not exist
        if not path.exists(path.dirname(file)):
            makedirs(path.dirname(file))
        # open file in binary write mode with buffering disabled
        self._fh = open(file, 'wb', 0)

    def __enter__(self):
        # make file connection and return it
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # make sure the file is closed
        self.close()

    @property
    def file_handle(self) -> BinaryIO:
        """The file handle to the binary file

        Returns
        -------
        BinaryIO
            The file handle.
        """
        return self._fh


    def close(self) -> None:
        """Close the file handle
        """
        if self._fh is not None:
            self._fh.close()

    def write_file(
        self,
        image_structure_function : ImageStructureFunction
        ) -> None:
        """Write image structure function to binary file

        Parameters
        ----------
        image_structure_function : ImageStructureFunction
            Image structure function object.
        """
        Nt, Ny, Nx = image_structure_function.shape
        Nextra = 2  # power spectrum + var
        dtype = 0   # at the moment we only have double precision 
        if image_structure_function.data.dtype == 'float64':
            dtype = 0
        elif image_structure_function.data.dtype == 'float32':
            dtype = 1
        self.write_header(Nt, Ny, Nx, Nextra, dtype)
        self.write_data(image_structure_function)

    def write_header(
        self,
        Nt : int,
        Ny : int,
        Nx : int,
        Nextra : int,
        dtype : int
        ) -> None:
        """Write image structure function file header

        Parameters
        ----------
        Nt : int
            Length of the image structure function array (i.e., number of lags)
        Ny : int
            Height of the image structure function array.
        Nx : int
            Width of the image structure function array.
        Nextra : int
            Number of extra data slices packed (i.e., variance and power spectrum).
        dtype : int
            Data type identifier. Possible values are:
            0 : float64
            1 : float32
        """
        offset = 0
        # write system endianness
        # 2 char, bytes 0-1
        if byteorder == 'little':
            self._fh.write('II'.encode("utf-8"))
        else:
            self._fh.write('MM'.encode("utf-8"))
        offset += 2 * format_byte_len['B']

        # write file identifier
        # unsigned short, bytes 2-3
        self._fh.write(struct.pack('H', 73))
        offset += format_byte_len['H']

        # write file version
        # 2 unsigned char, bytes 4-5
        self._fh.write(struct.pack('BB', *(self.version)))
        offset += 2 * format_byte_len['B']

        # write length of the image structure function data array
        # unsigned int, bytes 6-9
        self._fh.write(struct.pack('I', Nt))
        offset += format_byte_len['I']

        # write height of the image structure function data array
        # unsigned int, bytes 10-13
        self._fh.write(struct.pack('I', Ny))
        offset += format_byte_len['I']

        # write width of the image structure function data array
        # unsigned int, bytes 14-17
        self._fh.write(struct.pack('I', Nx))
        offset += format_byte_len['I']

        # write number of extra data slices packed (i.e., variance and power spectrum)
        # unsigned int, bytes 18-21
        self._fh.write(struct.pack('I', Nextra))
        offset += format_byte_len['I']

        # write dtype identifier
        # unsigned char, byte 22
        self._fh.write(struct.pack('B', dtype))
        offset += format_byte_len['B']

        # add empty bytes up to 63 for future use (if needed)
        self._fh.write(bytearray(HEADER_BYTE_LEN - offset))

    def write_data(
        self,
        image_structure_function : ImageStructureFunction
        ) -> None:
        """Write the data to binary file.

        Parameters
        ----------
        image_structure_function : ImageStructureFunction
            Image structure function object.
        """
        # FOR THE MOMENT WE ONLY HAVE DOUBLE (FLOAT64) VALUES
        Nt, Ny, Nx = image_structure_function.shape

        # write data
        for val in image_structure_function._data.flat:
            self._fh.write(struct.pack('d', val))
        data_offset = HEADER_BYTE_LEN
        extra_offset = data_offset + format_byte_len['d'] * Nt * Ny * Nx

        # write kx
        for kx in image_structure_function.kx:
            self._fh.write(struct.pack('d', kx))
        kx_offset = extra_offset + format_byte_len['d'] * 2 * Ny * Nx

        # write ky
        for ky in image_structure_function.ky:
            self._fh.write(struct.pack('d', ky))
        ky_offset = kx_offset + format_byte_len['d'] * Nx

        # write tau
        for tau in image_structure_function.tau:
            self._fh.write(struct.pack('d', tau))
        tau_offset = ky_offset + format_byte_len['d'] * Ny

        # write pixel size
        self._fh.write(struct.pack('d', image_structure_function.pixel_size))
        pixel_size_offset = tau_offset + format_byte_len['d'] * Nt

        # write delta t
        self._fh.write(struct.pack('d', image_structure_function.delta_t))
        delta_t_offset = pixel_size_offset + format_byte_len['d']

        # write offsets
        # extra offset
        self._fh.write(struct.pack('Q', extra_offset))
        # delta t offset
        self._fh.write(struct.pack('Q', delta_t_offset))
        # pixel size offset
        self._fh.write(struct.pack('Q', pixel_size_offset))
        # tau offset
        self._fh.write(struct.pack('Q', tau_offset))
        # ky offset
        self._fh.write(struct.pack('Q', ky_offset))
        # kx offset
        self._fh.write(struct.pack('Q', kx_offset))
        # data offset
        self._fh.write(struct.pack('Q', data_offset))
