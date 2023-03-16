# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""Functions to write binary fastddm image structure function file.
"""

import struct
from os import path
from typing import BinaryIO
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


class DdmSFReader(FramesSequenceND):
    