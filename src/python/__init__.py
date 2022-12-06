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
    **kwargs
) -> np.ndarray:
    backend: Dict[str, Dict[str, Callable]] = {
        "cpp": {"direct": dfm_direct_cpp, "fft": dfm_fft_cpp},
        "py": {
            "direct": partial(_py_image_structure_function, mode=mode),
            "fft": partial(_py_image_structure_function, mode=mode),
        },
    }

    # read actual image dimensions
    dim_t, dim_y, dim_x = img_seq.shape

    # dimensions after zero padding for efficiency and for normalization
    dim_x_padded = next_fast_len(dim_x)
    dim_y_padded = next_fast_len(dim_y)

    ddm_func = backend[core][mode]

    if core == "cpp":
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

        if mode == "fft":
            dim_t_padded = next_fast_len(2 * dim_t)
            args.append(dim_t_padded)

    else:
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

    return ddm_func(*args, **kwargs)
