# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

r"""Differential dynamic microscopy interface with backends.

The image structure function is computed from the images :math:`I(\vec{x}, t)`
as

.. math::

    D(\vec{q},\Delta t)=
    \langle \lvert \tilde{I}(\vec{q},t+\Delta t) -
    \tilde{I}(\vec{q},t) \rvert^2 \rangle_t

where :math:`\tilde{I}(\vec{q},t)` is the 2D Fourier transform of the image at
time :math:`t`.

The image power spectrum is computed as

.. math::

    \mathrm{PS}(\vec{q})=
    \langle\lvert\tilde{I}(\vec{q}, t)\rvert^2\rangle_t .

The background-corrected image power spectrum is computed as

.. math::

    \mathrm{VAR}(\vec{q})=
    \langle\lvert\tilde{I}(\vec{q},t)-\tilde{I}_0(\vec{q})\rvert^2\rangle_t=
    \mathrm{PS}(\vec{q}) - \lvert \tilde{I}_0 (\vec{q}) \rvert^2

where :math:`\tilde{I}_0(\vec{q}) = \langle\tilde{I}(\vec{q}, t)\rangle_t`.
This is just the variance over time of the 2D Fourier transformed image
sequence.
"""

from typing import Dict, Callable, Iterable, Optional
import warnings
from functools import partial
import numpy as np

from ._config import IS_CPP_ENABLED, IS_CUDA_ENABLED, DTYPE
from .imagestructurefunction import ImageStructureFunction
from ._ddm_python import _py_image_structure_function
from ._fftopt import next_fast_len

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
    window: Optional[np.ndarray] = None,
    **kwargs,
) -> ImageStructureFunction:
    """Perform Differential Dynamic Microscopy analysis on given image sequence.
    Returns the full image structure function.

    Parameters
    ----------
    img_seq : numpy.ndarray
        Image sequence of shape ``(t, y, x)`` where ``t`` is time.
    lags : Iterable[int]
        The delays to be inspected by the analysis.
    core : str, optional
        The backend core, choose between "py", "cpp", and "cuda".
        Default is "py".
    mode : str, optional
        The mode of calculating the structure function, choose between "diff"
        and "fft". Default is "fft".
    window : numpy.ndarray, optional
        A 2D array containing the window function to be applied to the images.
        Default is None.

    Returns
    -------
    ImageStructureFunction
        The image structure function.

    Raises
    ------
    RuntimeError
        If a value for ``core`` other than "py", "cpp", and "cuda" are given.
    RuntimeError
        If a value for ``mode`` other than "diff" and "fft" are given.
    RuntimeError
        If ``window`` and ``img_seq`` shapes are not compatible.
    RuntimeError
        If negative ``lags`` are given.
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

    # sanity check on lags
    # negative lags
    if np.min(lags) < 0:
        raise RuntimeError("Negative lags are not possible.")
    # large lags
    if np.max(lags) > dim_t - 1:  # maximum lag cna only go up to dim_t - 1 (included)
        raise RuntimeError(
            f"Lags larger than len(img_seq) - 1 = {dim_t - 1} are not possible."
        )
    # lag == 0
    # lags are sorted, 0 can only be in 0th position if no negative values are present
    if lags[0] == 0:
        warnings.warn(
            "Found 0 in lags. Removed for compatibility with other functions."
        )
        lags = lags[1:]

    # sanity check on window
    if window is None:
        window = np.array([], dtype=DTYPE)
    else:
        if window.shape != (dim_y, dim_x):
            raise RuntimeError(
                f"Window with shape {window.shape} incompatible with image shape {(dim_y, dim_x)}."
            )

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

        args.append(window.astype(DTYPE))

    else:
        args = [img_seq, lags, dim_x_padded, dim_y_padded, window.astype(DTYPE)]

    kx = 2 * np.pi * np.fft.fftfreq(dim_x_padded)[: (dim_x_padded // 2 + 1)]
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_y_padded))
    return ImageStructureFunction(
        ddm_func(*args, **kwargs),
        kx.astype(DTYPE),
        ky.astype(DTYPE),
        dim_x_padded,
        dim_y_padded,
        lags.astype(DTYPE),
    )
