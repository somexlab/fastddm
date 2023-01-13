"""Collection of functions to write and read binary files."""

import pickle
from pathlib import Path
from typing import Any
from os.path import dirname


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
    dir_name = Path(dir_name)
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
