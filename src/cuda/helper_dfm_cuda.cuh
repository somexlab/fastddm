// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __HELPER_DFM_CUDA_CUH__
#define __HELPER_DFM_CUDA_CUH__

/*! \file helper_dfm_cuda.cuh
    \brief Declaration of helper functions for Digital Fourier Microscopy on the GPU
*/

// *** headers ***
#include <cufft.h>
#define CUFFTCOMPLEX cufftDoubleComplex

// *** code ***

/*! \brief Convert array from T to double on device and prepare for fft2
    \param d_in     Input array
    \param d_out    Output array
    \param width    Width of the input array
    \param ipitch   Pitch of the input array
    \param idist    Distance between 2 consecutive elements of the input 3D array
    \param opitch   Pitch of the output array
    \param odist    Distance between 2 consecutive elements of the output 3D array
    \param N        Number of elements to copy/convert
 */
template <typename T>
__global__ void copy_convert_kernel(T *d_in,
                                    double *d_out,
                                    unsigned int width,
                                    unsigned int Npixels,
                                    unsigned int ipitch,
                                    unsigned int idist,
                                    unsigned int opitch,
                                    unsigned int odist,
                                    unsigned int N);

/*! \brief Scale array by constant -- b = A * a
\param a    input array
\param A    multiplication factor
\param b    output array
\param N    number of elements in the array
*/
__global__ void scale_array_kernel(double *a,
                                   double A,
                                   double *b,
                                   unsigned int N);

#endif