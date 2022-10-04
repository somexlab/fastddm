from math import log
from scipy import fft

def next_power_of_2(target):
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


def next_fast_len(target, fftw=False):
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

    Parameters
    ----------
    target : int
        Target value.
    fftw : bool, optional
        _description_, by default False

    Returns
    -------
    int
        Next fast input length.
    """

    # maximum value is next power of 2
    max_val = next_power_of_2(target)
    fast_len = 0

    if target == max_val:
        fast_len = max_val
    elif not fftw:
        fast_len = fft.next_fast_len(target)
    else:
        fast_len = max_val
        # find range of exponents
        N2 = target.bit_length()
        N3 = int(log(max_val)/log(3))+1
        N5 = int(log(max_val)/log(5))+1
        N7 = int(log(max_val)/log(7))+1

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
