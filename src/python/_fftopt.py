# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

from typing import List, Optional
import itertools
import math
import numpy as np
from scipy import fft

def next_power_of_2(target: int) -> int:
    """
    Return the next power of 2 greater than or equal to `x`.

    Parameters
    ----------
    target : int
        Target value.

    Returns
    -------
    int
        Next power of 2.
    """
    return 1<<(target-1).bit_length()


def next_fast_len(target: int, core: Optional[str]='py', force_even: Optional[bool]=False) -> int:
    """
    Returns the next fast size of input data to fft, for zero-padding.

    SciPy's FFT is efficient for small prime factors of the input length. Thus,
    the transforms are fastest when using composites of the prime factors
    handled by the fft implementation (i.e. :math:`2^a 3^b 5^c 7^d 11^e` where
    `e` is either 0 or 1).
    [See (Scipy documentation)[https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.next_fast_len.html#scipy.fft.next_fast_len]]

    FFTW is best at handling sizes of the form
    :math:`2^a 3^b 5^c 7^d 11^e 13^f` where :math:`e+f` is either 0 or 1.
    [See (FFTW documentation)[https://www.fftw.org/fftw3.pdf]]

    CUDA's cufft is best at handling sizes of the form
    :math:`2^a 3^b 5^c 7^d`.
    [See (cufft documentation)[https://docs.nvidia.com/cuda/cufft/index.html#introduction]]
    For very large transforms, though, the fastest varying dimension should be even.

    Parameters
    ----------
    target : int
        Target value.
    core : str, optional
        Select the backend ('py', 'cpp', 'cuda'). Default is 'py'.
    force_even : bool, optional
        Force even output. Only used if mode is 'cuda'. Default is False.

    Returns
    -------
    int
        Next fast input length.

    Raises
    ------
    ValueError
        If mode is not supported.
    """

    # maximum value is next power of 2
    max_val = next_power_of_2(target)

    if target == max_val:
        return max_val
    if core.upper() == 'PY':
        return fft.next_fast_len(target)
    if core.upper() == 'CPP':
        fast_len = max_val
        # find range of exponents
        N2 = target.bit_length()
        N3 = int(math.log(max_val)/math.log(3))+1
        N5 = int(math.log(max_val)/math.log(5))+1
        N7 = int(math.log(max_val)/math.log(7))+1

        for i in range(N2):
            pow2 = 2**i
            for j in range(N3):
                pow3 = 3**j
                for k in range(N5):
                    pow5 = 5**k
                    for w in range(N7):
                        pow7 = 7**w
                        curr_power = pow2 * pow3 * pow5 * pow7
                        if curr_power <= fast_len and curr_power >= target:
                            fast_len = curr_power
                            if fast_len == target:
                                break
                        curr_power = pow2 * pow3 * pow5 * pow7 * 11
                        if curr_power <= fast_len and curr_power >= target:
                            fast_len = curr_power
                            if fast_len == target:
                                break
                        curr_power = pow2 * pow3 * pow5 * pow7 * 13
                        if curr_power <= fast_len and curr_power >= target:
                            fast_len = curr_power
                            if fast_len == target:
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        return fast_len
    if core.upper() == 'CUDA':
        fast_len = max_val
        # find range of exponents
        N2 = target.bit_length()
        N3 = int(math.log(max_val)/math.log(3))+1
        N5 = int(math.log(max_val)/math.log(5))+1
        N7 = int(math.log(max_val)/math.log(7))+1

        start2 = 1 if force_even else 0

        for i in range(start2,N2):
            pow2 = 2**i
            for j in range(N3):
                pow3 = 3**j
                for k in range(N5):
                    pow5 = 5**k
                    for w in range(N7):
                        pow7 = 7**w
                        curr_power = pow2 * pow3 * pow5 * pow7
                        if curr_power <= fast_len and curr_power >= target:
                            fast_len = curr_power
                            if fast_len == target:
                                break
                    else:
                        continue
                    break
                else:
                    continue
                break
            else:
                continue
            break
        return fast_len

    raise ValueError('Mode not supported in `next_fast_len`.')


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
    sieve = np.ones(n//3 + (n%6 == 2), dtype=bool)
    for i in range(1, int(n**0.5)//3 + 1):
        if sieve[i]:
            k = 3*i + 1|1
            sieve[       k*k//3     ::2*k] = False
            sieve[k*(k-2*(i&1)+4)//3::2*k] = False
    return np.r_[2,3,((3*np.nonzero(sieve)[0][1:]+1)|1)]


def find_divisors(n: int) -> List[int]:
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
    primes = primesfrom2to(n+1).tolist()  # list of primes
    primes = map(int, primes)
    factors = {}
    for prime in primes:
        factor = 0
        while True:
            if n%prime == 0:
                factor += 1
                n /= prime
                factors[prime] = factor
            else: break

    powers = [
        [factor ** i for i in range(count + 1)]
        for factor, count in factors.items()
    ]

    divisors = [math.prod(i) for i in itertools.product(*powers)]
    divisors.sort()

    return divisors
