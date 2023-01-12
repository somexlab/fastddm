"""Collection of helper functions."""

from typing import Optional, Sequence

import numpy as np
import skimage.io as io
import tifffile


def tiff2numpy(path: str, seq: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read a TIFF file (or a sequence inside a multipage TIFF) and return it as a numpy array.

    Parameters
    ----------
    path : str
        The path to the TIFF file.
    seq : Sequence[int], optional
        A sequence, e.g. `range(5, 10)`, to describe a specific range within
        a multipage TIFF, by default None.

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


def images2numpy(fnames : Sequence[str]) -> np.ndarray:
    """Read a sequence of image files and return it as a numpy array.

    Parameters
    ----------
    fnames : Sequence[str]
        A sequence of file names.

    Returns
    -------
    np.ndarray
        The image sequence as a numpy array.

    Raises
    ------
    RuntimeError
        If color images are imported.
    """
    # open first image
    tmp = io.imread(fnames[0])
    if len(tmp.shape > 2):
        raise RuntimeError('Color images not supported.')

    # initialize image sequence array
    img_seq = np.zeros(shape=(len(fnames),*tmp.shape), dtype=tmp.dtype)

    # read images
    for i, f in enumerate(fnames):
        img_seq[i] = io.imread(f)
