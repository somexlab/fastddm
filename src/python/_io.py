# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.

"""Collection of functions to write and read binary files."""

import pickle
from pathlib import Path
from typing import Any, Sequence
from os.path import dirname
import numpy as np
import tifffile


def _store_data(
    data: Any,
    *,
    fname: str,
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Pickle any (picklable) data object as binary file with `name` under `path`.

    Parameters
    ----------
    data : Any
        A picklable data object.
    fname : str
        The name of the pickled data file.
    protocol : int, optional
        The pickle protocol version to be used, by default pickle.HIGHEST_PROTOCOL.
    """
    # ensure that storage folder exists
    dir_name = Path(dirname(fname))
    dir_name.mkdir(parents=True, exist_ok=True)

    with open(fname, "wb") as file:
        pickle.dump(data, file, protocol=protocol)


def load(fname: str) -> Any:
    """Read a pickled data object.

    Parameters
    ----------
    fname : str
        The path to the pickled data file.

    Returns
    -------
    Any
        The unpickled data.
    """
    fname = Path(fname)

    with open(fname, "rb") as file:
        data = pickle.load(file)

    return data


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
