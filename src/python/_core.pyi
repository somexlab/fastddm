# SPDX-FileCopyrightText: 2023-present University of Vienna
# SPDX-License-Identifier: GPL-3.0-or-later

"""Stub file for the compiled C++ extension module ``python._core``."""

from typing import List

import numpy as np

def ddm_diff(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    window: np.ndarray,
) -> np.ndarray: ...
def ddm_fft(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    nt: int,
    chunk_size: int,
    window: np.ndarray,
) -> np.ndarray: ...
