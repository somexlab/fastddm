"""Main function to interface with backends."""

import numpy as np
from typing import Iterable, Dict, Callable
from functools import partial

from ._dfm_python import _py_image_structure_function
from ._dfm_cpp import dfm_direct_cpp, dfm_fft_cpp
from ._fftopt import next_fast_len


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
