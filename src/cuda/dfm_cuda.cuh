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

/*! \brief Compute Image Structure Factor using the WK theorem on the GPU
    \param h_in         input array of Fourier transformed images
    \param lags         lags to be analyzed
    \param length       number of elements in z direction
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param nt           number of fft nodes in t direction
    \param num_chunks   number of q points chunks
    \param pitch_q      pitch of workspace1 device array (q-pitch, computed for complex elements)
    \param pitch_t      pitch of workspace2 device array (t-pitch, computed for complex elements)
    \param pitch_nt     pitch of workspace1 device array (nt-pitch, computed for complex elements)
 */
void correlate_fft(double *h_in,
                   vector<unsigned int> lags,
                   size_t length,
                   size_t nx,
                   size_t ny,
                   size_t nt,
                   size_t num_chunks,
                   size_t pitch_q,
                   size_t pitch_t,
                   size_t pitch_nt);

#endif