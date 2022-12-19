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
                                    unsigned long int width,
                                    unsigned long int Npixels,
                                    unsigned long int ipitch,
                                    unsigned long int idist,
                                    unsigned long int opitch,
                                    unsigned long int odist,
                                    unsigned long int N);

/*! \brief Scale array by constant -- b = A * a
\param a    input array
\param A    multiplication factor
\param b    output array
\param N    number of elements in the array
*/
__global__ void scale_array_kernel(double *a,
                                   double A,
                                   double *b,
                                   unsigned long int N);

/*! \brief Transpose complex matrix with pitch
    \param matIn    Input matrix
    \param ipitch   Pitch of input matrix
    \param matOut   Output matrix
    \param opitch   Pitch of output matrix
    \param width    Width of input matrix
    \param height   Height of input matrix
    \param NblocksX Number of blocks of tiles over x
    \param NblocksY Number of blocks of tiles over y
*/
__global__ void transpose_complex_matrix_kernel(double2 *matIn,
                                                unsigned long int ipitch,
                                                double2 *matOut,
                                                unsigned long int opitch,
                                                unsigned long int width,
                                                unsigned long int height,
                                                unsigned long int NblocksX,
                                                unsigned long int NblocksY);

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
                                                unsigned long int length,
                                                unsigned long int Nlags,
                                                unsigned long int Nq,
                                                unsigned long int Ntdt,
                                                unsigned long int pitch);

/*! \brief Make full power spectrum (copy symmetric part)
    \param d_in     input array
    \param ipitch   pitch of input array
    \param d_out    output array
    \param opitch   pitch of output array
    \param nxh      number of r2c fft elements over x
    \param nx       number of fft nodes over x
    \param ny       number of fft nodes over y
    \param N        number of 2d matrices
    \param NblocksX Number of blocks of tiles over x
    \param NblocksY Number of blocks of tiles over y
*/
__global__ void make_full_powspec_kernel(double2 *d_in,
                                         unsigned long int ipitch,
                                         double *d_out,
                                         unsigned long int opitch,
                                         unsigned long int nxh,
                                         unsigned long int nx,
                                         unsigned long int ny,
                                         unsigned long int N,
                                         unsigned long int NblocksX,
                                         unsigned long int NblocksY);

/*! \brief Shift power spectrum
    \param d_in     input array
    \param ipitch   pitch of input array
    \param d_out    output array
    \param opitch   pitch of output array
    \param nx       number of fft nodes over x
    \param ny       number of fft nodes over y
    \param N        number of 2d matrices
    \param NblocksX Number of blocks of tiles over x
    \param NblocksY Number of blocks of tiles over y
*/
__global__ void shift_powspec_kernel(double *d_in,
                                     unsigned long int ipitch,
                                     double *d_out,
                                     unsigned long int opitch,
                                     unsigned long int nx,
                                     unsigned long int ny,
                                     unsigned long int N,
                                     unsigned long int NblocksX,
                                     unsigned long int NblocksY);

/*! \brief Compute the square modulus of complex array
    \param d_in     Input complex array
    \param length   Number of elements in each subarray
    \param pitch    Distance between the first element of two consecutive subarrays
    \param N        Number of elements in the array
 */
__global__ void square_modulus_kernel(double2 *d_in,
                                      unsigned long int length,
                                      unsigned long int dist,
                                      unsigned long int N);

/*! \brief Copy real part of element into imaginary part of opposite element
    \param d_arr    Input complex array
    \param length   Number of elements in each subarray
    \param pitch    Distance (in number of elements) between the first element of two consecutive subarrays
    \param N        Total number of elements
 */
__global__ void real2imagopposite_kernel(double2 *d_arr,
                                         unsigned long int length,
                                         unsigned long int pitch,
                                         unsigned long int N);

/*! \brief Do final linear combination c[i] = (a[0] - b[i].x - 2 * a[i]) / (length - i)
    \param c        Output array
    \param pitch_c  Pitch of output array
    \param a        Input array 1 (fft correlation part)
    \param pitch_a  Pitch of input array a
    \param b        Input array 2 (cumsum part)
    \param pitch_b  Pitch of input array b
    \param length   Number of elements in each subarray
    \param N        Number of subarrays
*/
__global__ void linear_combination_final_kernel(double2 *c,
                                                unsigned long int pitch_c,
                                                double2 *a,
                                                unsigned long int pitch_a,
                                                double2 *b,
                                                unsigned long int pitch_b,
                                                unsigned long int length,
                                                unsigned long int N);

/*! \brief Keep only selected lags
    \param d_in     Input complex array
    \param d_out    Output complex array
    \param d_lags   Lags array
    \param Nlags    Number of lags
    \param ipitch   Pitch of input array
    \param opitch   Pitch of output array
    \param N        Number of subarrays
*/
__global__ void copy_selected_lags_kernel(double2 *d_in,
                                          double2 *d_out,
                                          unsigned int *d_lags,
                                          unsigned long int Nlags,
                                          unsigned long int ipitch,
                                          unsigned long int opitch,
                                          unsigned long int N);

#endif