"""Collection of helper functions."""

from typing import Optional, Sequence, Union, List

import numpy as np
import skimage.io as io
import tifffile
import psutil

from nd2reader import ND2Reader

def tiff2numpy(
    path: str,
    seq: Optional[Sequence[int]] = None,
    color_seq: Optional[Sequence[int]] = None
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
    else:
        _tiff_memchk(path=path, seq=seq, color_seq=color_seq)

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


def _tiff_memchk(
    path : str,
    seq: Optional[Sequence[int]] = None,
    color_seq: Optional[Sequence[int]] = None
) -> None:
    """Check if enough memory to read tiff file

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

    Raises
    ------
    RuntimeError
        If not enough memory.
    """
    # get available memory
    mem = psutil.virtual_memory().available

    with tifffile.TiffFile(path) as tif:
        dim_t = len(tif.pages)
        if seq is not None:
            dim_t = len(seq)
        tmp = tif.pages.get(0).asarray()
        if len(tmp.shape) == 3:
            dim_y, dim_x, dim_col = tmp.shape
            if color_seq is not None:
                dim_col = len(color_seq)
        else:
            dim_y, dim_x = tmp.shape
            dim_col = 1

        if mem < tmp.itemsize * dim_t * dim_y * dim_x * dim_col:
            raise RuntimeError("Not enough memory to read all images.")


def images2numpy(fnames : Sequence[str], color_seq: Optional[Sequence[int]] = None) -> np.ndarray:
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
    # get available memory
    mem = psutil.virtual_memory().available

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

        # check memory before reading
        if mem < tmp.itemsize * np.prod(shape):
            raise RuntimeError("Not enough memory to read all images.")
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images in c,t,y,x order
        for i, f in enumerate(fnames):
            tmp = io.imread(f)

            for j in color_seq:
                img_seq[j,i] = tmp[:,:,j]
    else:
        # initialize image sequence array
        shape = (len(fnames), *tmp.shape)

        # check memory before reading
        if mem < tmp.itemsize * np.prod(shape):
            raise RuntimeError("Not enough memory to read all images.")
        img_seq = np.zeros(shape=shape, dtype=tmp.dtype)

        # read images
        for i, f in enumerate(fnames):
            img_seq[i] = io.imread(f)

    return img_seq


def read_images(
    src: Union[str, List[str]],
    seq: Optional[Sequence[int]] = None,
    color_seq: Optional[Sequence[int]] = None
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
            return np.array(ND2Reader(src)[seq])
        else:
            return tiff2numpy(src, seq=seq, color_seq=color_seq)

    else:
        raise RuntimeError(f"Failed to open {src}.")
