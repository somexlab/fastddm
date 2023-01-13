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
        data = io.imread(path)
    else:
        # load the given image sequence with tifffile
        with tifffile.TiffFile(path) as tif:
            data = tif.asarray(key=seq)

    # check if images have also color channel
    if len(data.shape) > 3:
        data = data.transpose((3, 0, 1, 2))
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
        Coordinate convention is (z,y,x).
        If images have colors, convention is (c,z,y,x).
    """
    # open first image
    tmp = io.imread(fnames[0])
    if len(tmp.shape > 2):
        # initialize image sequence array
        shape = (tmp.shape[2], len(fnames), tmp.shape[:2])
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images in c,t,y,x order
        for i, f in enumerate(fnames):
            tmp = io.imread(f)
            for j in range(tmp.shape[2]):
                img_seq[j,i] = tmp[:,:,j]
    else:
        # initialize image sequence array
        shape = (len(fnames), *tmp.shape)
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images
        for i, f in enumerate(fnames):
            img_seq[i] = io.imread(f)
