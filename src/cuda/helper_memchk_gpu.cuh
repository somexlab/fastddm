// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __HELPER_MEMCHK_GPU_CUH__
#define __HELPER_MEMCHK_GPU_CUH__

/*! \file helper_memchk_gpu.cuh
    \brief Declaration of CUDA helper functions for memory check and optimization for GPU routines
*/

// *** headers ***
#include <cufft.h>

// *** code ***

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 16bytes elements
    \param N        subarray size
    \param pitch    pitch of the subarray
 */
void cudaGetDevicePitch16B(size_t N,
                           size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        subarray size
    \param pitch    pitch of the subarray
 */
void cudaGetDevicePitch8B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 4bytes elements
\param N        subarray size
\param pitch    pitch of the subarray
*/
void cudaGetDevicePitch4B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 2bytes elements
\param N        subarray size
\param pitch    pitch of the subarray
*/
void cudaGetDevicePitch2B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 1bytes elements
\param N        subarray size
\param pitch    pitch of the subarray
*/
void cudaGetDevicePitch1B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory size in bytes for fft2
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param batch        number of batch elements
    \param memsize      size (in bytes) of working area for fft2
    \param cufft_res    result of cufft function
 */
void cudaGetFft2MemSize(size_t nx,
                        size_t ny,
                        size_t batch,
                        size_t *memsize,
                        cufftResult &cufft_res);

/*! \brief Evaluate the device memory size in bytes for fft
    \param nt       number of fft nodes in t direction
    \param batch    number of batch elements
    \param pitch    pitch of input array
    \param memsize  size (in bytes) of working area for fft
    \param cufft_res    result of cufft function
*/
void cudaGetFftMemSize(size_t nt,
                       size_t batch,
                       size_t pitch,
                       size_t *memsize,
                       cufftResult &cufft_res);

#endif