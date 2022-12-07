from typing import List
import numpy as np

from ._memchk import get_free_mem
from ._gpumemchk import get_free_gpu_mem
from .core_cuda import get_device_pitch, get_device_fft2_mem, get_device_fft_mem
from .core_cuda import dfm_direct_cuda

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
    # if images are not double, get gpu x-pitch for buffer
    pitch_x = 0
    if not isinstance(img_seq[0,0,0], float):
        pitch_x = get_device_pitch(num_pixel_x, img_seq[0,0,0].itemsize)
    num_pixel_y = img_seq.shape[-2]
    # compute the number of iterations for fft2
    # give priority to number of host/device data transfer
    num_fft2 = 0
    while True:
        mem_gpu_req = 0
        num_fft2 += 1
        # compute number of batched fft2
        fft2_batch_len = (len(img_seq) - 1) // num_fft2 + 1
        # buffer -- pitch_x * Ny * fft2_batch_len * pixel_memsize
        if not isinstance(img_seq[0,0,0], float):
            mem_gpu_req += pitch_x * num_pixel_y * fft2_batch_len * pixel_memsize
        # workspace -- (nx // 2 + 1) * ny * fft2_batch_len * 2 * 8bytes
        mem_gpu_req += ((nx // 2) + 1) * ny * fft2_batch_len * 8 * 2
        # cufft2 internal buffer -- determined by `get_device_fft2_mem`
        mem_gpu_req += get_device_fft2_mem(nx, ny, fft2_batch_len)
        if mem_gpu > mem_gpu_req:
            break
        if num_fft2 == len(img_seq):
            raise MemoryError('Not enough space on GPU for fft2.')
    # compute number of q chunks
    # give priority to number of host/device data transfer
    num_chunks = 0

    # +++ ANALYZE +++
    return dfm_direct_cuda(img_seq, lags, nx, ny, num_fft2, pitch_x, num_chunks)
