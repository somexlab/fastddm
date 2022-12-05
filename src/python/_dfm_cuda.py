from typing import List
import numpy as np

from ._memchk import get_free_mem
from ._gpumemchk import get_free_gpu_mem

def dfm_direct_gpu(img_seq: np.ndarray, lags: List[int], nx: int, ny: int) -> np.ndarray:
    """Digital Fourier Microscopy, direct mode on GPU

    Compute the image structure function using differences on the GPU.

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

    # +++ CHECK MEMORY +++

    # get available memory on host
    mem = get_free_mem()
    mem_req = 0
    # calculations are done in double precision
    # we need:
    # output -- nx * ny * len(lags) * 8bytes
    mem_req += 8 * nx * ny * len(lags)
    # we require this space to be less than 90% of the available memory
    if int(0.9*mem) < mem_req:
        raise MemoryError('Not enough space. Cannot store result in memory.')

    # get available GPU memory
    mem_gpu = get_free_gpu_mem()[0]   # !! FOR NOW, I FIX THE GPU ID TO 0 !!
    # get memory size of pixel value
    pixel_memsize = img_seq[0,0,0].itemsize
    # get number of pixels over x and y in one image
    num_pixel_x = img_seq.shape[-1]
    num_pixel_y = img_seq.shape[-2]
    # compute the number of iterations for fft2
    # give priority to number of host/device data transfer
    num_fft2 = 0
    while True:
        mem_gpu_req = 0
        num_fft2 += 1
        # compute number of batched fft2
        fft2_batch_len = (len(img_seq) - 1) // num_fft2 + 1
        # buffer -- Nx * Ny * fft2_batch_len * pixel_memsize
        mem_gpu_req += num_pixel_x * num_pixel_y * fft2_batch_len * pixel_memsize
        # workspace -- (nx // 2 + 1) * ny * fft2_batch_len * 2 * 8bytes
        mem_gpu_req += ((nx // 2) + 1) * ny * fft2_batch_len * 8 * 2
        if mem_gpu > mem_gpu_req:
            break
        if num_fft2 == len(img_seq):
            raise MemoryError('Not enough space on GPU.')

    # +++ ANALYZE +++

    pass
