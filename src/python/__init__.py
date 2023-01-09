"""Main function to interface with backends."""

import numpy as np
from typing import Iterable, Dict, Callable, Union, Optional, Tuple
from functools import partial

from ._dfm_python import _py_image_structure_function
from ._dfm_cpp import dfm_direct_cpp, dfm_fft_cpp
from ._fftopt import next_fast_len

# range is a keyword argument to many functions, so save the builtin so they can
# use it.

def ddm(
    img_seq: np.ndarray,
    lags: Iterable[int],
    *,
    core: str = "py",
    mode: str = "fft",
    **kwargs,
) -> np.ndarray:
    """Perform DDM analysis on given image sequence. Returns the full image structure function.

    Parameters
    ----------
    img_seq : np.ndarray
        Image sequence of shape (t, y, x) where t is time.
    lags : Iterable[int]
        The delays to be inspected by the analysis.
    core : str, optional
        The backend core, choose between "py" and "cpp", by default "py"
    mode : str, optional
        The mode of calculating the autocorrelation, choose between "direct" and "fft", by default "fft"

    Returns
    -------
    np.ndarray
        The normalized image structure function.

    Raises
    ------
    RuntimeError
        If a value for `core` other than "py" and "cpp" are given.
    RuntimeError
        If a value for `mode` other than "direct" and "fft" are given.
    """
    # mapping core/mode strings to functors
    backend: Dict[str, Dict[str, Callable]] = {
        "cpp": {"direct": dfm_direct_cpp, "fft": dfm_fft_cpp},
        "py": {
            "direct": partial(_py_image_structure_function, mode=mode),
            "fft": partial(_py_image_structure_function, mode=mode),
        },
    }
    cores = list(backend.keys())
    modes = ["direct", "fft"]

    # sanity checks
    if core not in cores:
        raise RuntimeError(
            f"Unknown core '{core}' selected. Only possible options are {cores}."
        )

    if mode not in modes:
        raise RuntimeError(
            f"Unknown mode '{mode}' selected. Only possible options are {modes}."
        )

    # throw out duplicates and sort lags in ascending order
    lags = np.array(sorted(set(lags)))

    # read actual image dimensions
    dim_t, dim_y, dim_x = img_seq.shape

    # dimensions after zero padding for efficiency and for normalization
    fftw_flag = True if core == "cpp" else False
    dim_x_padded = next_fast_len(dim_x, fftw=fftw_flag)
    dim_y_padded = next_fast_len(dim_y, fftw=fftw_flag)

    # select backend
    ddm_func = backend[core][mode]

    # setup arguments
    if core == "cpp":
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

        if mode == "fft":
            dim_t_padded = next_fast_len(2 * dim_t, fftw=True)
            args.append(dim_t_padded)

    else:
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

    return ddm_func(*args, **kwargs)


def azimuthal_average(
    img_str_func : np.ndarray,
    kx : Optional[np.ndarray] = None,
    ky : Optional[np.ndarray] = None,
    bins : Optional[Union[int,Iterable[float]]] = 10,
    range : Optional[Tuple[float, float]] = None,
    mask : Optional[np.ndarray] = None,
    weights : Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the azimuthal average of the image structure function.
    Returns the azimuthal average and the bin centers.

    Parameters
    ----------
    img_str_func : np.ndarray
        The image structure function.
    kx : np.ndarray, optional
        The array of spatial frequencies along axis x. If kx is None,
        the frequencies evaluated with
        `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx))`
        are used (`Nx = img_str_func.shape[2]`). Default is None.
    ky : np.ndarray, optional
        The array of spatial frequencies along axis y. If ky is None,
        the frequencies evaluated with
        `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny))`
        are used (`Ny = img_str_func.shape[1]`). Default is None.
    bins : Union[int, Iterable[float]], optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10, by default). If `bins` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        `(k.min(), k.max())`, where `k` is the vector modulus computed from
        `kx` and `ky`. Values outside the range are ignored. The first element
        of the range must be less than or equal to the second.
    mask : np.ndarray, optional
        If a boolean `mask` is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of `img_str_func`.
    weights : np.ndarray, optional
        An array of weights, of the same y,x shape as `img_str_func`. Each
        value in `img_str_func` only contributes its associated weight towards
        the bin count (instead of 1).

    Returns
    -------
    az_avg : np.ndarray
        The azimuthal average.
    k : np.ndarray
        The spatial frequency associated to the bin.
    bin_edges : np.ndarray
        The bin edges.
    """

    # read actual image structure function shape
    dim_t, dim_y, dim_x = img_str_func.shape

    # check kx and ky
    if kx is None:
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_x))

    if ky is None:
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_y))

    # compute the k modulus
    X, Y = np.meshgrid(kx, ky)
    k_modulus = np.sqrt(X ** 2 + Y ** 2)

    # check range
    if range is None:
        k_min = np.min(k_modulus)
        k_max = np.max(k_modulus)
    else:
        k_min, k_max = range

    # check mask
    if mask is None:
        mask = np.full((dim_y, dim_x), True)

    # check weights
    if weights is None:
        weights = np.ones((dim_y, dim_x), dtype=np.float64)

    # compute bin edges and initialize k
    if isinstance(bins, int):
        bin_edges = np.linspace(k_min, k_max, bins)
        k = np.zeros(bins, dtype=np.float64)
    else:   # bins is an iterable
        bin_edges = [k_min]
        for bin_width in bins:
            bin_edges.append(bin_edges[-1] + bin_width)
        k = np.zeros(len(bins), dtype=np.float64)
        bins = len(bins) + 1

    # initialize azimuthal average
    az_avg = np.zeros((bins, dim_t), dtype=np.float64)

    # loop over bins
    for i, curr_bin_edge in enumerate(bin_edges):
        if i > 0:
            e_inf = bin_edges[i-1]
            e_sup = curr_bin_edge
            curr_px = (k_modulus > e_inf) & (k_modulus <= e_sup) & mask
        else:
            curr_px = (k_modulus == curr_bin_edge) & mask

        if np.all(np.logical_not(curr_px)):
            az_avg[i] = np.full(dim_t, np.nan)
            if i > 0:
                e_inf = bin_edges[i-1]
                e_sup = curr_bin_edge
                k[i] = (e_inf + e_sup) / 2.
            else:
                k[0] = curr_bin_edge
        else:
            num = (k_modulus[curr_px] * weights[curr_px]).mean()
            den = weights[curr_px].mean()
            k[i] = num / den
            w_avg = (img_str_func[:, curr_px] * weights[curr_px]).mean(axis=-1)
            az_avg[i] = w_avg / den

    return az_avg, k, bin_edges
