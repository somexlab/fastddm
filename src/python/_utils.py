"""Collection of helper functions.

Author: Fabian Krautgasser | fkrautgasser@posteo.org
"""

from typing import Optional, Sequence, Union, List

import numpy as np
import skimage.io as io
import tifffile

from nd2reader import ND2Reader


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
        The array containing the image information.
        Coordinate convention is (z,y,x).
        If color images are imported, convention is (c,z,y,x).
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


def images2numpy(fnames: Sequence[str]) -> np.ndarray:
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
        If color images are imported, convention is (c,z,y,x).
    """
    # open first image
    tmp = io.imread(fnames[0])
    if len(tmp.shape) > 2:
        # initialize image sequence array
        dim_y, dim_x, dim_col = tmp.shape
        shape = (dim_col, len(fnames), dim_y, dim_x)
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images in c,t,y,x order
        for i, f in enumerate(fnames):
            tmp = io.imread(f)
            for j in range(dim_col):
                img_seq[j, i] = tmp[:, :, j]
    else:
        # initialize image sequence array
        shape = (len(fnames), *tmp.shape)
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images
        for i, f in enumerate(fnames):
            img_seq[i] = io.imread(f)

    return img_seq


def read_images(
    src: Union[str, List[str]], seq: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Read a single image file or a list of image files.

    The single image file can itself be a sequence of images, like multipage tiff or nd2. If `seq`
    is given, e.g. with `seq=range(10)` or `seq=(10, 11, 12)` it will only be applied to single
    image files, again, like e.g. multipage tiff or nd2.

    Parameters
    ----------
    src : Union[str, List[str]]
        File path to a single image file or a list of file paths.
    seq : Optional[Sequence[int]], optional
        A subset of a multi-image file, can be set e.g. via a `range` object, by default None

    Returns
    -------
    np.ndarray
        The images in array form.

    Raises
    ------
    RuntimeError
        If the `src` type is not supported (e.g. generator expressions).
    """
    if isinstance(src, list):
        return images2numpy(src)

    elif isinstance(src, str):
        if src.endswith(".nd2"):
            return np.array(ND2Reader(src)[seq])
        else:
            return tiff2numpy(src, seq=seq)

    else:
        raise RuntimeError(f"Failed to open {src}.")


def chunkify(data: np.ndarray, chunksize: int, overlap: int = 0) -> List[np.ndarray]:
    """akes a dataset (or dataset indices) and chunks it into smaller portions of size
    `chunksize`, with a given `overlap` with the previous chunk.

    The last chunk may not be of the right size. The chunking will happen along the __first__ axis.

    Parameters
    ----------
    data : np.ndarray
        A numpy array of to be chunked content
    chunksize : int
        The size of the output chunks.
    overlap : int, optional
        Give a number > 0 by how much the chunks should overlap, i.e. overlap=2 would result in [[1 2 3], [2 3 4], [3 4 5], ...], by default 0

    Returns
    -------
    List[np.ndarray]
        The list of chunks.
    """
    size = len(data)
    nchunks = size // chunksize
    if nchunks == 0:  # nothing to do here
        return [data]

    left, right, diff = 0, chunksize, chunksize - overlap
    chunks = []

    # main chunks
    while right < size:
        chunks.append(data[left:right])
        left += diff
        right += diff

    # rest chunk if any
    if len(data[left:]) > 0:
        chunks.append(data[left:])

    return chunks
