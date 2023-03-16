# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""Functions to write binary fastddm image structure function file.
"""

import struct
from os import path
from typing import BinaryIO, Tuple
from .imagestructurefunction import ImageStructureFunction
from pims.base_frames import FramesSequenceND, Frame
import numpy as np


format_byte_len = {
    'B' : 1,    # unsigned char
    'H' : 2,    # unsigned short
    'I' : 4,    # unsigned int
    'Q' : 8,    # unsigned long long
    'f' : 4,    # float
    'd' : 8,    # double
}

dtype2format = {
    0 : 'd',
    1 : 'f',
}


class DdmSFReader(FramesSequenceND):
    """PIMS wrapper for the fastddm image structure function
    binary file parser. Use this class to process your .sf.ddm files.
    """

    def __init__(self, file : str):
        super(DdmSFReader, self).__init__()

        if not file.endswith('.sf.ddm'):
            raise ValueError("Invalid file name extension. Expected .sf.ddm extension.")
        self.filename = file
        self._fh = open(file, "rb")

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


class SFParser:
    """Parse fastddm image structure function file.
    """

    supported_file_versions = {(0, 1) : True}

    def __init__(self, fh):
        self._fh = fh
        self.metadata = None

        # check if the file version is supported
        self.supported = self._check_version_supported()

        # parse metadata
        self._parse_metadata()

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

    def get_version(self) -> Tuple[int, int]:
        """Determines version of the .sf.ddm file

        Returns
        -------
        Tuple[int, int]
            Major and minor version.
        """
        # the version starts after 4 bytes
        self._fh.seek(4)

        # the next two bytes contain the major and minor version packed as 2 unsigned char
        data = self._fh.read(2)
        major_version, minor_version = struct.unpack("BB", data)

        return major_version, minor_version
    
    def _parse_metadata(self) -> None:
        """Reads metadata and creates dictionary.
        """
        metadata = {}
        self._fh.seek(0)

        # endianness is at bytes 0-1, encoded as utf-8
        # 'II' : little-endian
        # 'MM' : big-endian
        endianness = self._fh.read(2).decode("utf-8")
        if endianness == 'II':
            metadata['byteorder'] = 'little'
        elif endianness == 'MM':
            metadata['byteorder'] = 'big'
        else:
            raise ValueError(f"Unsupported endianness {endianness} identifier.")
        
        # length of image structure function data array
        # is at bytes 6-9, packed as unsigned int (I)
        self._fh.seek(6)
        data = self._fh.read(format_byte_len['I'])
        metadata['Nt'] = struct.unpack('I', data)

        # height of image structure function data array
        # is at bytes 10-13, packed as unsigned int (I)
        data = self._fh.read(format_byte_len['I'])
        metadata['Ny'] = struct.unpack('I', data)

        # width of image structure function data array
        # is at bytes 14-17, packed as unsigned int (I)
        data = self._fh.read(format_byte_len['I'])
        metadata['Nx'] = struct.unpack('I', data)

        # number of extra slices in image structure function data array
        # (i.e., variance and power spectrum)
        # is at bytes 18-21, packed as unsigned int (I)
        data = self._fh.read(format_byte_len['I'])
        metadata['Nextra'] = struct.unpack('I', data)

        # dtype identifier
        # is at byte 22, packed as unsigned char
        data = self._fh.read(format_byte_len['B'])
        metadata['dtype'] = int(struct.unpack('B', data))

        # data byte offset
        self._fh.seek(-format_byte_len['Q'], 2)
        data = self._fh.read(format_byte_len['Q'])
        metadata['data_offset'] = struct.unpack('Q', data)

        # kx byte offset
        self._fh.seek(-2 * format_byte_len['Q'], 2)
        data = self._fh.read(format_byte_len['Q'])
        metadata['kx_offset'] = struct.unpack('Q', data)

        # ky byte offset
        self._fh.seek(-3 * format_byte_len['Q'], 2)
        data = self._fh.read(format_byte_len['Q'])
        metadata['ky_offset'] = struct.unpack('Q', data)

        # tau byte offset
        self._fh.seek(-4 * format_byte_len['Q'], 2)
        data = self._fh.read(format_byte_len['Q'])
        metadata['tau_offset'] = struct.unpack('Q', data)

        # pixel size byte offset
        self._fh.seek(-5 * format_byte_len['Q'], 2)
        data = self._fh.read(format_byte_len['Q'])
        pixel_size_offset = struct.unpack('Q', data)

        # delta_t byte offset
        self._fh.seek(-6 * format_byte_len['Q'], 2)
        data = self._fh.read(format_byte_len['Q'])
        delta_t_offset = struct.unpack('Q', data)

        # extra byte offset
        self._fh.seek(-7 * format_byte_len['Q'], 2)
        data = self._fh.read(format_byte_len['Q'])
        metadata['extra_offset'] = struct.unpack('Q', data)

        # pixel_size
        self._fh.seek(pixel_size_offset)
        data = self._fh.read(dtype2format[metadata['dtype']])
        metadata['pixel_size'] = struct.unpack(dtype2format[metadata['dtype']], data)

        # delta_t
        self._fh.seek(delta_t_offset)
        data = self._fh.read(dtype2format[metadata['dtype']])
        metadata['delta_t'] = struct.unpack(dtype2format[metadata['dtype']], data)

        self.metadata = metadata

    def get_image(self, index : int) -> Frame:
        """Creates a Frame object and returns the
        image structure function slice at the selected index.

        Parameters
        ----------
        index : int
            Index.

        Returns
        -------
        Frame
            The image structure function Frame.
        """

        frame_offset = self._calculate_frame_offset(index)

    def _calculate_frame_offset(self, index : int):
