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
    # --- ESTIMATE HOST MEMORY REQUIRED FOR FFT2
    # calculations are done in double precision
    # we need:
    # output -- nx * ny * len(lags) * 8bytes
    # mem_req += 8 * nx * ny * len(lags)
    # but to store the intermediate fft2, we need:
    # fft2 -- 2 * (nx // 2 + 1) * ny * len(img_seq) * 8bytes
    # which is always larger than the output size (output is then resized)
    mem_req += 8 * 2 * (nx // 2 + 1) * ny * len(img_seq)
    # --- ESTIMATE HOST MEMORY REQUIRED FOR STRUCTURE FUNCTION PART
    # helper array of t1 (unsigned int, 32 bits)
    # t1 -- (len(img_seq) - lags[0]) * len(lags) * 4bytes   (AT MOST!)
    # helper array of num (unsigned int, 32 bits)
    # num -- (len(img_seq) - lags[0]) * 4bytes
    mem_req += 4 * (len(img_seq) * lags[0]) * (len(lags) + 1)

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
    # --- ESTIMATE DEVICE MEMORY REQUIRED FOR FFT2
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
    # --- ESTIMATE DEVICE MEMORY REQUIRED FOR STRUCTURE FUNCTION PART
    # compute number of q chunks
    # give priority to number of host/device data transfer
    num_chunks = 0
    pitch_t = get_device_pitch(2 * len(img_seq), 2 * 8)     # 8 is for double
    while True:
        # helper array of t1 (unsigned int, 32 bits)
        # t1 -- (len(img_seq) - lags[0]) * len(lags) * 4bytes   (AT MOST!)
        # helper array of num (unsigned int, 32 bits)
        # num -- (len(img_seq) - lags[0]) * 4bytes
        mem_gpu_req = 4 * (len(img_seq) * lags[0]) * (len(lags) + 1)
        # compute number of batched q vectors
        num_chunks += 1
        chunk_size = ((nx//2 + 1) * ny - 1) // num_chunks + 1
        pitch_q = get_device_pitch(2 * chunk_size, 2 * 8)   # 8 is for double
        # workspace1 -- ((chunk_size+pitch_q-1)//pitch_q)*pitch_q *
        #             * ((len(img_seq)+pitch_t-1)//pitch_t)*pitch_t *
        #             * 2 * 8bytes
        ws1_size = ((chunk_size + pitch_q - 1) // pitch_q) * pitch_q
        ws1_size *= ((len(img_seq) + pitch_t - 1) // pitch_t) * pitch_t
        ws1_size *= 2 * 8
        # workspace2 is same as workspace1
        mem_gpu_req += 2 * ws1_size
        if mem_gpu > mem_gpu_req:
            break
        if num_chunks == ((nx//2 + 1) * ny):
            raise MemoryError('Not enough space on GPU for correlation.')

    # +++ ANALYZE +++
    return dfm_direct_cuda(img_seq, lags, nx, ny, num_fft2, pitch_x, num_chunks)
