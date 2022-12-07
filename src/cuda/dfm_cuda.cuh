// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __DFM_CUDA_CUH__
#define __DFM_CUDA_CUH__

/*! \file dfm_cuda.cuh
    \brief Declaration of core CUDA Digital Fourier Microscopy functions
*/

// *** headers ***
#include <vector>

using namespace std;

// *** code ***

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
    \param nx       number of fft nodes in x direction
    \param ny       number of fft nodes in y direction
    \param batch    number of batch elements
    \param memsize  size (in bytes) of working area for fft2
 */
void cudaGetFft2MemSize(size_t nx,
                        size_t ny,
                        size_t batch,
                        size_t *memsize);

/*! \brief Evaluate the device memory size in bytes for fft
    \param nt       number of fft nodes in t direction
    \param batch    number of batch elements
    \param pitch    pitch of input array
    \param memsize  size (in bytes) of working area for fft
*/
void cudaGetFftMemSize(size_t nt,
                       size_t batch,
                       size_t pitch,
                       size_t *memsize);

#endif