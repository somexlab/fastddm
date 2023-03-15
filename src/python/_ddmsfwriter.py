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
        self.offset = 0
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

    def write_file(self, image_structure_function : ImageStructureFunction) -> None:
        Nt, Ny, Nx = image_structure_function.shape
        Nextra = 2  # power spectrum + var
        dtype = 0   # WE DON'T HAVE FOR THE MOMENT A SINGLE PRECISION OPTION, SO WE FIX DOUBLE FOR NOW
        self.write_header(Nt, Ny, Nx, Nextra, dtype)

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
        self.offset = 0
        # write system endianness
        # 2 char, bytes 0-1
        if byteorder == 'little':
            self._fh.write('II'.encode("utf-8"))
        else:
            self._fh.write('MM'.encode("utf-8"))
        self.offset += 2 * format_byte_len['B']

        # write file identifier
        # unsigned short, bytes 2-3
        self._fh.write(struct.pack('H', 73))
        self.offset += format_byte_len['H']

        # write file version
        # 2 unsigned char, bytes 4-5
        self._fh.write(struct.pack('BB', *(self.version)))
        self.offset += 2 * format_byte_len['B']

        # write length of the image structure function data array
        # unsigned int, bytes 6-9
        self._fh.write(struct.pack('I', Nt))
        self.offset += format_byte_len['I']

        # write height of the image structure function data array
        # unsigned int, bytes 10-13
        self._fh.write(struct.pack('I', Ny))
        self.offset += format_byte_len['I']

        # write width of the image structure function data array
        # unsigned int, bytes 14-17
        self._fh.write(struct.pack('I', Nx))
        self.offset += format_byte_len['I']

        # write number of extra data slices packed (i.e., variance and power spectrum)
        # unsigned int, bytes 18-21
        self._fh.write(struct.pack('I', Nextra))
        self.offset += format_byte_len['I']

        # write dtype identifier
        # unsigned char, byte 22
        self._fh.write(struct.pack('B', dtype))
        self.offset += format_byte_len['B']
