"""Collection of helper functions."""

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import skimage.io as io
import tifffile


def tif_to_numpy(path: Path, seq: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read a TIFF file (or a sequence inside a multipage TIFF) and return it as a numpy array.

    Parameters
    ----------
    path : Path
        The path to the TIFF file.
    seq : Optional[Sequence[int]], optional
        A sequence, e.g. `range(5, 10)`, to describe a specific range within a multipage TIFF, by default None

    Returns
    -------
    np.ndarray
        The array containing the image information; coordinate convention is (z,y,x).
    """
    if seq is None:
        return io.imread(path)

    # load the given image sequence with tifffile
    with tifffile.TiffFile(path) as tif:
        data = tif.asarray(key=seq)
    return data
