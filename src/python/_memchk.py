# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

import psutil

def get_free_mem() -> int:
    """Get available memory

    Returns
    -------
    int
        Available memory on RAM.
    """
    return psutil.virtual_memory().available
