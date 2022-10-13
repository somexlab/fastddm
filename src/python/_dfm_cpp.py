from .core import dfm_direct, dfm_fft

import psutil


def dfm_direct_cpp(img_seq, lags, nx, ny):
    """Digital Fourier Microscopy, direct mode

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

    Returns
    -------
    isf : numpy.ndarray
        Image structure function.

    Raises
    ------
    MemoryError
        If memory is not sufficient to perform the calculations.
    """

    # get available memory
    mem = psutil.virtual_memory().available
    mem_req = 0
    # calculations are done in double precision
    # we need:
    #  workspace -- 2 * (nx/2 + 1) * ny * len(img_seq) * 4bytes
    mem_req += 4 * (2 * (nx//2 + 1) * ny * len(img_seq))
    #  tmp -------- len(lags) * 4bytes
    mem_req += 4 * len(lags)
    # we require this space to be less than 80% of the available memory
    # to stay on the safe side
    if int(0.8*mem) < mem_req:
        raise MemoryError('Not enough memory')

    return dfm_direct(img_seq, lags, nx, ny)


def dfm_fft_cpp(img_seq, lags, nx, ny, nt):
    """Digital Fourier Microscopy, fft mode

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

    Returns
    -------
    isf : numpy.ndarray
        Image structure function.

    Raises
    ------
    MemoryError
        If memory is not sufficient to perform the calculations.
    """

    # get available memory
    mem = psutil.virtual_memory().available

    # compute the optimal bundle size
    rep = 1
    while True:
        # if the number of repetitions is larger than the number
        # of 2D Fourier nodes, memory is not enough
        if rep > (nx//2 + 1)*ny:
            raise MemoryError('Not enough memory')

        # evaluate bundle size (is rep*bundle_size == (nx//2+1) * ny ? )
        if ((nx//2 + 1)*ny) % rep == 0:
            bundle_size = ((nx//2 + 1)*ny) // rep
        else:
            rep += 1
            continue

        mem_req = 0
        # calculations are done in double precision
        # we need:
        #  workspace1 -- 2 * (nx/2 + 1) * ny * len(img_seq) * 4bytes
        mem_req += 4 * (2 * (nx//2 + 1) * ny * len(img_seq))
        #  workspace2 -- 2 * bundle_size * nt * 4bytes
        mem_req += 4 * (2 * bundle_size * nt)
        #  tmp --------- bundle_size * 4bytes
        mem_req += 4 * bundle_size
        # we require this space to be less than 80% of the available memory
        # to stay on the safe side

        if int(0.8*mem) < mem_req:
            rep += 1
        else:
            break

    return dfm_fft(img_seq, lags, nx, ny, nt, bundle_size)
