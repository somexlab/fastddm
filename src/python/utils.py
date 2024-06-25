# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Fabian Krautgasser
# Maintainer: Fabian Krautgasser

"""Collection of helper functions."""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union
from enum import Enum
import warnings

import numpy as np
import skimage.io as io
import tifffile
from nd2reader import ND2Reader

# custom types & constants
Metadata = Dict[str, Any]
OUTPUT_ORDER = "CTYX"


def tiff2numpy(
    src: str,
    seq: Optional[Sequence[int]] = None,
    color_seq: Optional[Sequence[int]] = None,
    input_order: Optional[str] = None,
) -> np.ndarray:
    """Read a TIFF file (or a sequence inside a multipage TIFF) and return it as a numpy array.

    The tiff file is assumed to be of the shape (T,Y,X). If ``color_seq`` is given, the order
    (C,T,Y,X) is assumed, otherwise the option ``input_order`` needs to be specified. If
    ``input_order`` is given, the full tiff file is read into memory, the axes ordered accordingly,
    and then ``seq`` and ``color_seq`` applied (if specified).

    If a non-tiff file is given, it is opened with ``skimage.io.imread()`` using only default
    parameters.

    It will be reshaped into (T,Y,X) (or (C,T,Y,X) if color channels are present) before it is
    returned.

    Parameters
    ----------
    src : str
        The path to the TIFF file.
    seq : Sequence[int], optional
        A sequence, e.g. ``range(5, 10)``, to describe a specific range within
        a multipage TIFF, by default None.
    color_seq : Sequence[int], optional
        A sequence, e.g. ``range(2)``, to describe a specific color sequence to be selected, by
        default None.
    input_order : str, optional
        The order of input dimensions. Currently only supports up to 4 dimensions, ``"CTYX"``,
        by default None.

    Returns
    -------
    numpy.ndarray
        The array containing the image information.
        Coordinate convention is (T,Y,X) (or (C,T,Y,X)).
    """
    if not src.endswith(".tif"):  # read anything but tif files with io.imread
        warnings.warn(
            "Non-tiff file, returning array opened with default settings only."
        )
        return io.imread(src)

    if input_order is not None:  # read whole array first
        # read whole array first
        data = tifffile.imread(src)

        # input sanity check
        if len(input_order) != len(data.shape):
            raise RuntimeError(
                f"Given input dimensions '{input_order}' and loaded data shape "
                f"{data.shape} mismatch:"
            )

        if not all([dim in OUTPUT_ORDER for dim in list(input_order)]):
            raise RuntimeError(
                f"Unrecognized dimension in '{input_order}'. Only allowed values"
                f" are '{OUTPUT_ORDER}'."
            )

        # enum to match the actual number of dimensions
        Order = Enum(
            "axes",
            [dim for dim in list(OUTPUT_ORDER) if dim in input_order],
            start=0,
        )
        transpose_mapper = tuple(Order[dim].value for dim in input_order)
        data = np.transpose(data, axes=transpose_mapper)  # unify output

        if "C" in input_order:
            if color_seq is not None:
                data = data[color_seq, ...]  # slice color channels

            if seq is not None:
                data = data[:, seq, ...]  # slice time dimension

        elif seq is not None:
            data = data[seq, ...]

        return data

    if color_seq is not None:
        data = tifffile.imread(src, key=color_seq)  # here we already assume CTYX order

        if seq is not None:
            data = data[:, seq, ...]

    elif seq is not None:
        data = tifffile.imread(src, key=seq)  # here we assume TYX order

    else:
        data = tifffile.imread(src)

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
        default None.

    Returns
    -------
    numpy.ndarray
        The image sequence as a numpy array.
        Coordinate convention is ``(z,y,x)``.
        If color images are imported, convention is ``(c,z,y,x)``.
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
        A (sub)sequence of images to be selected from the movie, e.g. ``list(range(1, 200))``,
        by default None.

    Returns
    -------
    numpy.ndarray
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
    input_order: Optional[str] = None,
) -> np.ndarray:
    """Read a single image file or a list of image files.

    The single image file can itself be a sequence of images, like multipage tiff or nd2. If ``seq``
    is given, e.g. with ``seq=range(10)`` or ``seq=(10, 11, 12)`` it will only be applied to single
    image files, again, like e.g. multipage tiff or nd2.

    Parameters
    ----------
    src : Union[str, List[str]]
        File path to a single image file or a list of file paths.
    seq : Optional[Sequence[int]], optional
        A subset of a multi-image file, can be set e.g. via a ``range`` object, by default None.
    color_seq : Sequence[int], optional
        A sequence, e.g. ``range(2)``, to describe a specific color sequence to be selected, by
        default None.
    input_order : str, optional
        The order of input dimensions. Currently only supports up to 4 dimensions ``"CTYX"``,
        only used for TIFF files, be default None.

    Returns
    -------
    numpy.ndarray
        The images in array form.

    Raises
    ------
    RuntimeError
        If the ``src`` type is not supported (e.g. generator expressions).
    """
    if isinstance(src, list):
        return images2numpy(src, color_seq=color_seq)

    elif isinstance(src, str):
        if src.endswith(".nd2"):
            return _read_nd2(src, seq=seq)

        else:
            return tiff2numpy(
                src, seq=seq, color_seq=color_seq, input_order=input_order
            )

    else:
        raise RuntimeError(f"Failed to open {src}.")


def _read_tiff_metadata(src: str) -> Metadata:
    """Reads the raw metadata of the first page of a tiff file and returns it as a dictionary.

    Parameters
    ----------
    src : str
        Path to TIFF file.

    Returns
    -------
    Metadata
        Raw metadata of first tiff page.
    """
    metadata = {}
    with tifffile.TiffFile(src) as tif:
        for tag in tif.pages[0].tags:
            metadata[tag.name] = tag.value

    return metadata


def read_metadata(src: str) -> Metadata:
    """Reads an images metadata and returns it as a dictionary.

    Currently only supports .nd2 files and raw metadata for tiff files.

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
        ".nd2": lambda s: ND2Reader(s).metadata,
        ".tif": lambda s: _read_tiff_metadata(s),
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
    """Takes a sequence ``seq`` and chunks it into smaller portions of size ``chunksize``, with a
    given ``overlap`` with the previous chunk.

    The sequence could be e.g. image indices, or an image sequence itself. However, be aware that
    in the latter case, depending on the chunksize and overlap settings the needed amount of memory
    could be very high! (It is recommended to use image sequence indices, see example below.)

    The last chunk may not be of the right size. The chunking will happen along the *first* axis.


    Parameters
    ----------
    seq : numpy.ndarray
        A numpy array of to be chunked content.
    chunksize : int
        The size of the output chunks.
    overlap : int, optional
        Give a number > 0 by how much the chunks should overlap, i.e. ``overlap=2`` would result in
        ``[[1 2 3], [2 3 4], [3 4 5], ...]``, by default 0.

    Returns
    -------
    List[numpy.ndarray]
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
