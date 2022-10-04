import numpy as np

def logspace_int(stop,num=50,endpoint=False):
    """
    Return quasi-evenly log-spaced integers over a specified interval.

    Returns `num` evenly spaced samples, calculated over the interval `[1, stop]`.

    The endpoint of the interval can optionally be included.

    Parameters
    ----------
    stop : int
        The end value of the sequence, unless `endpoint` is set to False.
        In that case, the sequence consists of all but the last of
        `num + 1` samples, so that `stop` is excluded.
        Note that the step size changes when `endpoint` is False.
    num : int, optional
        Number of samples to generate. Default is 50. Must be > 0.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -------
    samples : ndarray
        There are `num` equally spaced samples in the closed interval
        `[1, stop]` or the half-open interval `[1, stop)` (depending on whether
        `endpoint` is True or False).

    See Also
    --------
    fibonacci : Returns the Fibonacci sequence.

    Examples
    --------
    >>> a = logspace_int(10, num=5)
    array([1, 2, 3, 6, 10])
    >>> a = logspace_int(10, num=5, endpoint=False)
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

    for i in range(1,len(samples)):
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


def fibonacci(stop,endpoint=False):
    """
    Return fibonacci sequence over a specified interval.

    Returns fibonacci samples, calculated over the interval `[1, stop]`.

    The endpoint of the interval can optionally be included.

    Parameters
    ----------
    stop : int
        The end value of the sequence, unless `endpoint` is set to False.
    endpoint : bool, optional
        If True, `stop` is the last sample. Otherwise, it is not included.
        Default is False.

    Returns
    -------
    samples : ndarray
        Fibonacci samples in the closed interval `[1, stop]` or the half-open
        interval `[1, stop)` (depending on whether `endpoint` is True or False).

    See Also
    --------
    logspace_int : Returns quasi-evenly log-spaced integers over a specified interval.

    Examples
    --------
    >>> a = fibonacci(13)
    array([1, 2, 3, 5, 8, 13])
    >>> a = fibonacci(13, endpoint=False)
    array([1, 2, 3, 5, 8])

    """

    # initialize list
    samples = [1,2]

    if not endpoint:
        stop = stop - 1

    # evaluate samples
    while True:
        new_sample = samples[-2]+samples[-1]
        if new_sample <= stop:
            samples.append(new_sample)
        else:
            break

    # make numpy array from list
    samples = np.array(samples, dtype=int)

    return samples
