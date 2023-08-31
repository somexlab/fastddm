# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""The collection of C++ functions to perform Differential Dynamic Microscopy."""

from typing import List, Optional
import itertools
import math
import psutil
import numpy as np

from ._core import ddm_diff, ddm_fft
from ._config import IS_SINGLE_PRECISION, DTYPE

SCALAR_SIZE = 4 if IS_SINGLE_PRECISION else 8


def ddm_diff_cpp(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    window: np.ndarray,
    **kwargs
):
    """Differential Dynamic Microscopy, diff mode

    Compute the image structure function using differences.

    Parameters
    ----------
    img_seq : numpy.ndarray
        Input image sequence.
    lags : array_like
        List of selected lags.
    nx : int
        Number of fft nodes in x direction.
    ny : int
        Number of fft nodes in y direction.
    window : np.ndarray
        A 2D array containing the window function to be applied to the images.
        If window is empty, no window is applied.
    Returns
    -------
    np.ndarray
        The half-plane image structure function.

    Raises
    ------
    RuntimeError
        If memory is not sufficient to perform the calculations.
    """
    # get available memory
    mem = psutil.virtual_memory().available
    mem_req = 0
    # calculations are done in double precision
    # we need:
    #  workspace -- 2 * (nx/2 + 1) * ny * max(len(img_seq), len(lags) + 2) * (SCALAR_SIZE)bytes
    mem_req += SCALAR_SIZE * (2 * (nx // 2 + 1) * ny * max(len(img_seq), len(lags) + 2))
    #  tmp, tmp2 -------- [len(lags) + 3] * 8bytes
    mem_req += 8 * (len(lags) + 3)
    # we require this space to be less than 90% of the available memory
    # to stay on the safe side
    if int(0.9 * mem) < mem_req:
        raise RuntimeError("Not enough memory")

    return ddm_diff(img_seq, lags, nx, ny, window)


def ddm_fft_cpp(
    img_seq: np.ndarray,
    lags: List[int],
    nx: int,
    ny: int,
    nt: int,
    window: np.ndarray,
    **kwargs
):
    """Differential Dynamic Microscopy, fft mode

    Compute the image structure function using the Wiener-Khinchin theorem.

    Parameters
    ----------
    img_seq : numpy.ndarray
        Input image sequence.
    lags : array_like
        List of selected lags.
    nx : int
        Number of fft nodes in x direction.
    ny : int
        Number of fft nodes in y direction.
    nt : int
        Number of fft nodes in t direction.
    window : np.ndarray
        A 2D array containing the window function to be applied to the images.
        If window is empty, no window is applied.

    Returns
    -------
    np.ndarray
        The half-plane image structure function.

    Raises
    ------
    RuntimeError
        If memory available is not sufficient for calculations.
    """
    # get available memory
    mem = psutil.virtual_memory().available

    # compute the optimal chunk size
    divisors = find_divisors((nx // 2 + 1) * ny)
    idx = len(divisors)
    while True:
        idx -= 1
        chunk_size = divisors[idx]

        mem_req = 0
        # calculations are done in double precision
        # we need:
        #  workspace1 -- 2 * (nx/2 + 1) * ny * max(len(img_seq), len(lags)+2) * (SCALAR_SIZE)bytes
        mem_req += SCALAR_SIZE * (
            2 * (nx // 2 + 1) * ny * max(len(img_seq), len(lags) + 2)
        )
        #  workspace2 -- 2 * chunk_size * nt * 8 bytes
        mem_req += 8 * (2 * chunk_size * nt)
        #  tmp --------- chunk_size * 8 bytes
        mem_req += 8 * chunk_size
        #  tmpAvg --------- chunk_size * 8 bytes
        mem_req += 8 * chunk_size
        # we require this space to be less than 90% of the available memory
        # to stay on the safe side
        if int(0.9 * mem) > mem_req:
            break
        if idx == 0:
            raise RuntimeError("Not enough memory")

    return ddm_fft(img_seq, lags, nx, ny, nt, chunk_size, window)


def primesfrom2to(n: int) -> List[int]:
    """Returns a list of primes, 2 <= p < n

    Parameters
    ----------
    n : int
        Upper limit value.

    Returns
    -------
    List[int]
        Primes up to n (excluded).
    """
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=bool)
    for i in range(1, int(n**0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[k * k // 3 :: 2 * k] = False
            sieve[k * (k - 2 * (i & 1) + 4) // 3 :: 2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0][1:] + 1) | 1)]


def find_divisors(n):
    """Find the divisors of n

    Parameters
    ----------
    n : int
        Input value.

    Returns
    -------
    divisors : List[int]
        List of divisors of n.
    """
    primes = primesfrom2to(n + 1).tolist()  # list of primes
    primes = map(int, primes)
    factors = {}
    for prime in primes:
        factor = 0
        while True:
            if n % prime == 0:
                factor += 1
                n /= prime
                factors[prime] = factor
            else:
                break

    powers = [
        [factor**i for i in range(count + 1)] for factor, count in factors.items()
    ]

    divisors = [math.prod(i) for i in itertools.product(*powers)]
    divisors.sort()

    return divisors
