// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __DDM_CUDA_CUH__
#define __DDM_CUDA_CUH__

/*! \file ddm_cuda.cuh
    \brief Declaration of core CUDA Differential Dynamic Microscopy functions
*/

// *** headers ***
#include <vector>

using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Transfer images on GPU and compute fft2
    \param h_in         input array
    \param h_out        output array
    \param h_window     window function
    \param is_window    True if window function is given (not empty)
    \param width        width of input array
    \param height       height of input array
    \param length       number of elements in z direction
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param num_fft2     number of fft2 chunks
    \param buff_pitch   pitch of buffer device array
    \param pitch_nx     pitch of output fft2 array (for complex values)
 */
template <typename T>
void compute_fft2(const T *h_in,
                  Scalar *h_out,
                  const Scalar *h_window,
                  bool is_window,
                  unsigned long long width,
                  unsigned long long height,
                  unsigned long long length,
                  unsigned long long nx,
                  unsigned long long ny,
                  unsigned long long num_fft2,
                  unsigned long long buff_pitch,
                  unsigned long long pitch_nx);

/*! \brief Compute image structure function using differences on the GPU
    \param h_in         input array of Fourier transformed images
    \param lags         lags to be analyzed
    \param length       number of elements in z direction
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param num_chunks   number of q points chunks
    \param pitch_q      pitch of device array (q-pitch)
    \param pitch_t      pitch of device array (t-pitch)
 */
void structure_function_diff(Scalar *h_in,
                             vector<unsigned int> lags,
                             unsigned long long length,
                             unsigned long long nx,
                             unsigned long long ny,
                             unsigned long long num_chunks,
                             unsigned long long pitch_q,
                             unsigned long long pitch_t);

/*! \brief Convert to fftshifted image structure function on the GPU
    \param h_in             input array after structure function calculation
    \param Nlags            number of lags analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param num_shift        number of shift chunks
    \param pitch_fs         pitch of device array for shift operations
 */
void make_shift(Scalar *h_in,
                unsigned long long Nlags,
                unsigned long long nx,
                unsigned long long ny,
                unsigned long long num_shift,
                unsigned long long pitch_fs);

/*! \brief Compute image structure function using the WK theorem on the GPU
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
void structure_function_fft(Scalar *h_in,
                            vector<unsigned int> lags,
                            unsigned long long length,
                            unsigned long long nx,
                            unsigned long long ny,
                            unsigned long long nt,
                            unsigned long long num_chunks,
                            unsigned long long pitch_q,
                            unsigned long long pitch_t,
                            unsigned long long pitch_nt);

#endif // __DDM_CUDA_CUH__
