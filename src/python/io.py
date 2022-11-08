"""Collection of functions to write and read binary files."""

import pickle
from pathlib import Path
from typing import Any


def store_data(
    data: Any,
    *,
    path: Path,
    name: str = "analysis-blob",
    protocol: int = pickle.HIGHEST_PROTOCOL,
) -> None:
    """Pickle any (picklable) data object as binary file with `name` under `path`.

    Parameters
    ----------
    data : Any
        A picklable data object.
    path : Path
        The folder to contain the pickled data.
    name : str, optional
        The name of the pickled data file (automatically appends ".pkl" file ending), by default "analysis-blob"
    protocol : int, optional
        The pickle protocol version to be used, by default pickle.HIGHEST_PROTOCOL
    """
    # ensure that storage folder exists
    path.mkdir(parents=True, exist_ok=True)

    # check file ending
    name = name if name.endswith(".pkl") else f"{name}.pkl"

    with open(path / name, "wb") as file:
        pickle.dump(data, file, protocol=protocol)


def read_data(path: Path) -> Any:
    """Read a pickled data object.

    Parameters
    ----------
    path : Path
        The path to the pickled data file.

    Returns
    -------
    Any
        The unpickled data.
    """
    with open(path, "rb") as file:
        data = pickle.load(file)

    return data
