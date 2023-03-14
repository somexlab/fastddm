# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Fabian Krautgasser
# Maintainer: Fabian Krautgasser

"""Collection of helper functions."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import skimage.io as io
import tifffile
from nd2reader import ND2Reader

# custom types
Metadata = Dict[str, Any]


def tiff2numpy(
    path: str,
    seq: Optional[Sequence[int]] = None,
    color_seq: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """Read a TIFF file (or a sequence inside a multipage TIFF) and return it as a numpy array.

    If the tiff file contains a color channel, the axis order is assumed to be (T,Y,X,C) while
    reading the file; it will be reshaped into (C,T,Y,X) before it is returned.

    Parameters
    ----------
    path : str
        The path to the TIFF file.
    seq : Sequence[int], optional
        A sequence, e.g. `range(5, 10)`, to describe a specific range within
        a multipage TIFF, by default None.
    color_seq : Sequence[int], optional
        A sequence, e.g. `range(2)`, to describe a specific color sequence to be selected, by
        default None

    Returns
    -------
    np.ndarray
        The array containing the image information.
        Coordinate convention is (Z,Y,X) (or (T,Y,X))
        If color images are imported, convention is (C,Z,Y,X).
    """
    if not path.endswith(".tif"):  # read anything but tif files with io.imread
        return io.imread(path)

    with tifffile.TiffFile(path) as tif:
        data = tif.asarray(key=seq)

        if tif.pages.is_multipage:
            if len(data.shape) == 4:
                data = np.transpose(data, axes=(3, 0, 1, 2))
        elif len(data.shape) == 3:
            data = np.transpose(data, axes=(2, 0, 1))

    if color_seq is not None and len(data.shape) >= 3:
        data = data[color_seq, ...]

    return data


def images2numpy(
    fnames: Sequence[str], color_seq: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Read a sequence of image files and return it as a numpy array.

    Parameters
    ----------
    fnames : Sequence[str]
        A sequence of file names.
    color_seq : Sequence[int], optional
        A sequence, e.g. `range(2)`, to describe a specific color sequence to be selected, by
        default None

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

        # check if color sequence is given
        if color_seq is not None:
            color_seq = tuple(color_seq)

            # set dim_col to new value
            dim_col = len(color_seq)
        else:
            color_seq = range(dim_col)  # all color channels

        shape = (dim_col, len(fnames), dim_y, dim_x)
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images in c,t,y,x order
        for i, f in enumerate(fnames):
            tmp = io.imread(f)

            for j in color_seq:
                img_seq[j, i] = tmp[:, :, j]
    else:
        # initialize image sequence array
        shape = (len(fnames), *tmp.shape)
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images
        for i, f in enumerate(fnames):
            img_seq[i] = io.imread(f)

    return img_seq


def _read_nd2(src: str, seq: Optional[Sequence[int]] = None) -> np.ndarray:
    """Read an ND2 file with an optional specific (sub)sequence of images.

    Parameters
    ----------
    src : str
        Path to ND2 file.
    seq : Optional[Sequence[int]], optional
        A (sub)sequence of images to be selected from the movie, e.g. list(range(1, 200)), by default None

    Returns
    -------
    np.ndarray
        The image sequence as numpy array.
    """

    mov = ND2Reader(src)
    length, dim_y, dim_x = mov.shape  # what about color channels? 
    length = len(seq) if seq is not None else length
    dtype = mov[0].dtype  # access dtype of first frame
    imgs = np.zeros((length, dim_y, dim_x), dtype=dtype)

    selected = seq if seq is not None else range(length)
    for i, idx in enumerate(selected):
        frame = mov.get_frame(idx)
        imgs[i] = frame.data

    return imgs


def read_images(
    src: Union[str, List[str]],
    seq: Optional[Sequence[int]] = None,
    color_seq: Optional[Sequence[int]] = None,
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
    color_seq : Sequence[int], optional
        A sequence, e.g. `range(2)`, to describe a specific color sequence to be selected, by
        default None
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
        return images2numpy(src, color_seq=color_seq)

    elif isinstance(src, str):
        if src.endswith(".nd2"):
            return _read_nd2(src, seq=seq)
        
        else:
            return tiff2numpy(src, seq=seq, color_seq=color_seq)

    else:
        raise RuntimeError(f"Failed to open {src}.")


def read_metadata(src: str) -> Metadata:
    """Reads an images metadata and returns it as a dictionary.

    Currently only supports .nd2 files.

    Parameters
    ----------
    src : str
        Path to file location.

    Returns
    -------
    Metadata
        A dictionary of all available metadata (no content checking is performed).

    Raises
    ------
    NotImplementedError
        If a folder is given as argument.
    FileNotFoundError
        If the given file does not exist.
    RuntimeError
        For non-supported image file types.
    """
    type_reader: Dict[str, Callable[[str], Metadata]] = {
        ".nd2": lambda s: ND2Reader(s).metadata
    }
    supported_types = type_reader.keys()  # for readability

    path = Path(src)
    if path.is_dir():
        raise NotImplementedError(
            "It is not yet supported to read the metadata of multiple images in a directory."
        )

    elif not path.exists():
        raise FileNotFoundError(f"Given file '{src}' does not exist.")

    elif path.suffix not in supported_types:
        raise RuntimeError(f"Given file extension '{path.suffix}' is not supported.")

    return type_reader[path.suffix](src)


def chunkify(seq: np.ndarray, chunksize: int, overlap: int = 0) -> List[np.ndarray]:
    """Takes a sequence `seq` and chunks it into smaller portions of size `chunksize`, with a given
    `overlap` with the previous chunk.

    The sequence could be e.g. image indices, or an image sequence itself. However, be aware that
    in the latter case, depending on the chunksize & overlap settings the needed amount of memory
    could be very high! (It is recommended to use image sequence indices, see example below.)

    The last chunk may not be of the right size. The chunking will happen along the __first__ axis.


    Parameters
    ----------
    seq : np.ndarray
        A numpy array of to be chunked content
    chunksize : int
        The size of the output chunks.
    overlap : int, optional
        Give a number > 0 by how much the chunks should overlap, i.e. overlap=2 would result in [[1 2 3], [2 3 4], [3 4 5], ...], by default 0

    Returns
    -------
    List[np.ndarray]
        The list of chunks.

    Examples
    --------
    >>> chunkify(np.arange(10), chunksize=5, overlap=2)
    [array([0, 1, 2, 3, 4]), array([3, 4, 5, 6, 7]), array([6, 7, 8, 9])]
    """
    size = len(seq)
    nchunks = size // chunksize
    if nchunks == 0:  # nothing to do here
        return [seq]

    left, right, diff = 0, chunksize, chunksize - overlap
    chunks = []

    # main chunks
    while right < size:
        chunks.append(seq[left:right])
        left += diff
        right += diff

    # rest chunk if any
    if len(seq[left:]) > 0:
        chunks.append(seq[left:])

    return chunks
