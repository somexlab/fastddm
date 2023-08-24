# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Enrico Lattuada
# Maintainer: Enrico Lattuada

"""This module contains the functions for lag time array creation.

Lag time arrays can be used to speed-up the computation of the
image structure function (when computed using the differences
scheme) or to reduce its size:

.. code-block:: python

    import fastddm as fddm

    img_seq = ...   # load your images here

    # use an array of quasi logspaced int indices
    lags = fddm.lags.logspace_int(len(img_seq), num=100)

    dqt = fddm.ddm(img_seq, lags)


They can also be used to resample an azimuthal average:

.. code-block:: python

    # compute azimuthal average aa
    # resample the azimuthal average
    # use a fibonacci array of delay times 
    new_taus = fddm.lags.fibonacci(len(aa)) * aa.tau[0]
    aa_res = aa.resample(new_taus)
"""

import numpy as np


def logspace_int(stop: int,
                 num: int = 50,
                 endpoint: bool = False) -> np.ndarray:
    """
    Return quasi-evenly log-spaced integers over a specified interval.

    Returns ``num`` evenly spaced samples, calculated over the interval ``[1, stop]``.
    The endpoint of the interval can optionally be included.

    Parameters
    ----------
    stop : int
        The end value of the sequence, unless ``endpoint`` is False.
        In that case, the sequence consists of all but the last of
        ``num + 1`` samples, so that ``stop`` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be > 0.
    endpoint : bool, optional
        If True, ``stop`` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -------
    numpy.ndarray
        The ``num`` log-spaced samples.

    Examples
    --------
    >>> a = logspace_int(10, num=5, endpoint=True)
    array([1, 2, 3, 6, 10])
    >>> a = logspace_int(10, num=5)
    array([1, 2, 3, 4, 7])
    """
    # initialize array
    samples = np.empty(shape=num)

    # compute ratio
    if endpoint:
        ratio = (stop)**(1./num)
    else:
        ratio = (stop)**(1./(num+1))

    if not endpoint:
        num += 1

    # evaluate samples
    samples[0] = 1

    for i in range(1, len(samples)):
        # compute next value
        next_val = samples[i-1] * ratio

        # check round
        if next_val - samples[i-1] >= 1.:
            samples[i] = next_val
        else:
            # force +1
            samples[i] = samples[i-1] + 1
            # update the ratio
            ratio = (stop/samples[i])**(1/(num-i-1))

    # now round the values to integers
    samples = np.round(samples).astype(int)

    return samples


def fibonacci(stop: int,
              endpoint: bool = False) -> np.ndarray:
    """
    Return fibonacci sequence over a specified interval.

    Returns fibonacci samples, calculated over the interval ``[1, stop]``.

    The endpoint of the interval can optionally be included.

    Parameters
    ----------
    stop : int
        The end value of the sequence, unless ``endpoint`` is False.
    endpoint : bool, optional
        If True, ``stop`` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -------
    numpy.ndarray
        Fibonacci samples.

    Examples
    --------
    >>> a = fibonacci(13, endpoint=True)
    array([1, 2, 3, 5, 8, 13])
    >>> a = fibonacci(13)
    array([1, 2, 3, 5, 8])
    """
    # initialize list
    samples = [1, 2]

    if not endpoint:
        stop = stop - 1

    # evaluate samples
    while True:
        new_sample = samples[-2] + samples[-1]
        if new_sample <= stop:
            samples.append(new_sample)
        else:
            break

    # make numpy array from list
    samples = np.array(samples, dtype=int)

    return samples
