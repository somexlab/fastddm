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

/*! \brief Transpose complex matrix with pitch
    \param matIn    Input matrix
    \param ipitch   Pitch of input matrix
    \param matOut   Output matrix
    \param opitch   Pitch of output matrix
    \param width    Width of input matrix
    \param height   Height of input matrix
*/
__global__ void transpose_complex_matrix_kernel(double2 *matIn,
                                                unsigned int ipitch,
                                                double2 *matOut,
                                                unsigned int opitch,
                                                unsigned int width,
                                                unsigned int height);

/*! \brief Compute correlation using differences
    \param d_in     input array
    \param d_out    output array
    \param d_lags   array of lags
    \param d_t1     array of starting times t1
    \param d_num    array of number of previous occurrences
    \param length   length
    \param Nlags    number of lags
    \param Nq       number of q values (chunk size)
    \param Ntdt     length of d_t1
    \param pitch    pitch of arrays
*/
__global__ void correlatewithdifferences_kernel(double2 *d_in,
                                                double2 *d_out,
                                                unsigned int *d_lags,
                                                unsigned int *d_t1,
                                                unsigned int *d_num,
                                                unsigned int length,
                                                unsigned int Nlags,
                                                unsigned int Nq,
                                                unsigned int Ntdt,
                                                unsigned int pitch);

/*! \brief Make full power spectrum (copy symmetric part)
    \param d_in     input array
    \param ipitch   pitch of input array
    \param d_out    output array
    \param opitch   pitch of output array
    \param nxh      number of r2c fft elements over x
    \param nx       number of fft nodes over x
    \param ny       number of fft nodes over y
    \param N        number of 2d matrices
*/
__global__ void make_full_powspec_kernel(double2 *d_in,
                                         unsigned int ipitch,
                                         double *d_out,
                                         unsigned int opitch,
                                         unsigned int nxh,
                                         unsigned int nx,
                                         unsigned int ny,
                                         unsigned int N);

/*! \brief Shift power spectrum
    \param d_in     input array
    \param ipitch   pitch of input array
    \param d_out    output array
    \param opitch   pitch of output array
    \param nx       number of fft nodes over x
    \param ny       number of fft nodes over y
    \param N        number of 2d matrices
*/
__global__ void shift_powspec_kernel(double *d_in,
                                     unsigned int ipitch,
                                     double *d_out,
                                     unsigned int opitch,
                                     unsigned int nx,
                                     unsigned int ny,
                                     unsigned int N);

#endif