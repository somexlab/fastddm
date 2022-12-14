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

/*! \brief Transfer images on GPU and compute fft2
    \param h_in         input array
    \param h_out        output array
    \param width        width of input array
    \param height       height of input array
    \param length       number of elements in z direction
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param num_fft2     number of fft2 chunks
    \param buff_pitch   pitch of buffer device array
 */
template <typename T>
void compute_fft2(const T *h_in,
                  double *h_out,
                  size_t width,
                  size_t height,
                  size_t length,
                  size_t nx,
                  size_t ny,
                  size_t num_fft2,
                  size_t buff_pitch);

/*! \brief Compute Image Structure Function using differences on the GPU
    \param h_in         input array of Fourier transformed images
    \param lags         lags to be analyzed
    \param length       number of elements in z direction
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param num_chunks   number of q points chunks
    \param pitch_q      pitch of device array (q-pitch)
    \param pitch_t      pitch of device array (t-pitch)
 */
void correlate_direct(double *h_in,
                      vector<unsigned int> lags,
                      size_t length,
                      size_t nx,
                      size_t ny,
                      size_t num_chunks,
                      size_t pitch_q,
                      size_t pitch_t);

/*! \brief Convert to full and fftshifted Image Structure Function on the GPU
    \param h_in             input array after structure function calculation
    \param lags             lags to be analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param num_fullshift    number of full and shift chunks
    \param pitch_fs         pitch of device array for full and shift operations
 */
void make_full_shift(double *h_in,
                     vector<unsigned int> lags,
                     size_t nx,
                     size_t ny,
                     size_t num_fullshift,
                     size_t pitch_fs);

#endif