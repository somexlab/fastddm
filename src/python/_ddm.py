# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

"""Differential dynamic microscopy interface with backends."""

from typing import Dict, Callable, Iterable
from functools import partial
import numpy as np

from fastddm import IS_CPP_ENABLED, IS_CUDA_ENABLED
from fastddm.imagestructurefunction import ImageStructureFunction
from fastddm._ddm_python import _py_image_structure_function
from fastddm._fftopt import next_fast_len

## setup of backend dictionary, initially only with py backend
_backend: Dict[str, Dict[str, Callable]] = {
    "py": {
        "diff": partial(_py_image_structure_function, mode="diff"),
        "fft": partial(_py_image_structure_function, mode="fft"),
    }
}

if IS_CPP_ENABLED:
    from fastddm._ddm_cpp import ddm_diff_cpp, ddm_fft_cpp
    # enable cpp support in backends
    _backend["cpp"] = {
        "diff": ddm_diff_cpp,
        "fft": ddm_fft_cpp,
    }

if IS_CUDA_ENABLED:
    from fastddm._ddm_cuda import ddm_diff_gpu, ddm_fft_gpu
    # enable cuda support in backends
    _backend["cuda"] = {
        "diff": ddm_diff_gpu,
        "fft": ddm_fft_gpu,
    }


def ddm(
    img_seq: np.ndarray,
    lags: Iterable[int],
    *,
    core: str = "py",
    mode: str = "fft",
    **kwargs,
    ) -> ImageStructureFunction:
    """Perform Differential Dynamic Microscopy analysis on given image sequence.
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
        The mode of calculating the autocorrelation, choose between "diff"
        and "fft". Default is "fft".

    Returns
    -------
    ImageStructureFunction
        The image structure function.

    Raises
    ------
    RuntimeError
        If a value for `core` other than "py", "cpp", and "cuda" are given.
    RuntimeError
        If a value for `mode` other than "diff" and "fft" are given.
    """

    # renaming for convenience
    backend = _backend
    cores = list(backend.keys())
    modes = ["diff", "fft"]

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

    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_x_padded))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_y_padded))
    return ImageStructureFunction(ddm_func(*args, **kwargs), kx, ky, lags.astype(np.float64))