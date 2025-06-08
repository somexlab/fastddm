# SPDX-FileCopyrightText: 2023-present University of Vienna
# SPDX-FileCopyrightText: 2023-present Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino
# SPDX-License-Identifier: GPL-3.0-or-later

# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

"""Collection of functions to write and read binary files."""

from typing import Any

from .azimuthalaverage import AAReader
from .imagestructurefunction import SFReader
from .intermediatescatteringfunction import ISFReader


def load(fname: str) -> Any:
    """Read a binary data object.

    Parameters
    ----------
    fname : str
        The path to the binary data file.

    Returns
    -------
    Any
        The loaded object.
    """
    if fname.endswith(".sf.ddm"):
        with SFReader(fname) as f:
            return f.load()
    if fname.endswith(".aa.ddm"):
        with AAReader(fname) as f:
            return f.load()
    if fname.endswith(".isf.ddm"):
        with ISFReader(fname) as f:
            return f.load()
    raise RuntimeError("File extension not recognized.")
