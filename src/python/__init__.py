"""Main function to interface with backends."""

import numpy as np
from typing import Iterable, Dict, Callable, Optional
from functools import partial

IS_CPP_ENABLED = ${IS_CPP_ENABLED}      # configured by CMake
IS_CUDA_ENABLED = ${IS_CUDA_ENABLED}    # configured by CMake

from ._dfm_python import _py_image_structure_function

## setup of backend dictionary, initially only with py backend
_backend: Dict[str, Dict[str, Callable]] = {
    "py": {
        "direct": partial(_py_image_structure_function, mode="direct"),
        "fft": partial(_py_image_structure_function, mode="fft"),
    }
}

from ._fftopt import next_fast_len

if IS_CPP_ENABLED:
    from ._dfm_cpp import dfm_direct_cpp, dfm_fft_cpp
    # enable cpp support in backends
    _backend["cpp"] = {
        "direct": dfm_direct_cpp,
        "fft": dfm_fft_cpp
    }

if IS_CUDA_ENABLED:
    from ._dfm_cuda import dfm_direct_gpu, dfm_fft_gpu
    # enable cuda support in backends
    _backend["cuda"] = {
        "direct": dfm_direct_gpu,
        "fft": dfm_fft_gpu
    }

def ddm(
    img_seq: np.ndarray,
    lags: Iterable[int],
    *,
    core: str = "py",
    mode: str = "fft",
    **kwargs,
) -> np.ndarray:
    """Perform DDM analysis on given image sequence.
    Returns the full image structure function.

    Parameters
    ----------
    img_seq : np.ndarray
        Image sequence of shape (t, y, x) where t is time.
    lags : Iterable[int]
        The delays to be inspected by the analysis.
    core : str, optional
        The backend core, choose between "py", "cpp", and "cuda".
        Default is "py".
    mode : str, optional
        The mode of calculating the autocorrelation, choose between "direct"
        and "fft". Default is "fft".

    Returns
    -------
    np.ndarray
        The normalized image structure function.

    Raises
    ------
    RuntimeError
        If a value for `core` other than "py", "cpp", and "cuda" are given.
    RuntimeError
        If a value for `mode` other than "direct" and "fft" are given.
    """

    # renaming for convenience
    backend = _backend
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
    dim_x_padded = next_fast_len(dim_x, core)
    dim_y_padded = next_fast_len(dim_y, core)

    # select backend
    ddm_func = backend[core][mode]

    # setup arguments
    if core != "py":
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

        if mode == "fft":
            dim_t_padded = next_fast_len(2 * dim_t, core)
            args.append(dim_t_padded)

    else:
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

    return ddm_func(*args, **kwargs)
