// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __HELPER_DDM_CUDA_CUH__
#define __HELPER_DDM_CUDA_CUH__

/*! \file helper_ddm_cuda.cuh
    \brief Declaration of helper functions for Differential Dynamic Microscopy on the GPU
*/

// *** headers ***
#include <cufft.h>
#define CUFFTCOMPLEX cufftDoubleComplex

// *** code ***

/*! \brief Convert array from T to double on device and prepare for fft2
    \param d_in     Input array
    \param d_out    Output array
    \param width    Width of the input array
    \param height   Height of the input array
    \param length   Length of the input array
    \param ipitch   Pitch of the input array
    \param idist    Distance between 2 consecutive elements of the input 3D array
    \param opitch   Pitch of the output array
    \param odist    Distance between 2 consecutive elements of the output 3D array
 */
template <typename T>
__global__ void copy_convert_kernel(T *d_in,
                                    double *d_out,
                                    unsigned long long width,
                                    unsigned long long height,
                                    unsigned long long length,
                                    unsigned long long ipitch,
                                    unsigned long long idist,
                                    unsigned long long opitch,
                                    unsigned long long odist);

/*! \brief Scale array by constant -- A * a
    \param a        input array
    \param pitch    pitch of input array
    \param length   length of each subarray
    \param A        multiplication factor
    \param N        number of subarrays
 */
__global__ void scale_array_kernel(double2 *a,
                                   unsigned long long pitch,
                                   unsigned long long length,
                                   double A,
                                   unsigned long long N);

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
                                                unsigned long long ipitch,
                                                double2 *matOut,
                                                unsigned long long opitch,
                                                unsigned long long width,
                                                unsigned long long height,
                                                unsigned long long NblocksX,
                                                unsigned long long NblocksY);

/*! \brief Compute structure function using differences
    \param d_in     input array
    \param d_out    output array
    \param d_lags   array of lags
    \param length   length
    \param Nlags    number of lags
    \param Nq       number of q values (chunk size)
    \param pitch    pitch of arrays
*/
__global__ void structure_function_diff_kernel(double2 *d_in,
                                               double2 *d_out,
                                               unsigned int *d_lags,
                                               unsigned long long length,
                                               unsigned long long Nlags,
                                               unsigned long long Nq,
                                               unsigned long long pitch);

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
                                         unsigned long long ipitch,
                                         double *d_out,
                                         unsigned long long opitch,
                                         unsigned long long nxh,
                                         unsigned long long nx,
                                         unsigned long long ny,
                                         unsigned long long N,
                                         unsigned long long NblocksX,
                                         unsigned long long NblocksY);

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
                                     unsigned long long ipitch,
                                     double *d_out,
                                     unsigned long long opitch,
                                     unsigned long long nx,
                                     unsigned long long ny,
                                     unsigned long long N,
                                     unsigned long long NblocksX,
                                     unsigned long long NblocksY);

/*! \brief Compute the square modulus of complex array
    \param d_in     Input complex array
    \param length   Number of elements in each subarray
    \param pitch    Distance between the first element of two consecutive subarrays
    \param N        Number of subarrays
 */
__global__ void square_modulus_kernel(double2 *d_in,
                                      unsigned long long length,
                                      unsigned long long pitch,
                                      unsigned long long N);

/*! \brief Copy real part of element into imaginary part of opposite element
    \param d_arr    Input complex array
    \param length   Number of elements in each subarray
    \param pitch    Distance (in number of elements) between the first element of two consecutive subarrays
    \param N        Number of subarrays
 */
__global__ void real2imagopposite_kernel(double2 *d_arr,
                                         unsigned long long length,
                                         unsigned long long pitch,
                                         unsigned long long N);

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
                                                unsigned long long pitch_c,
                                                double2 *a,
                                                unsigned long long pitch_a,
                                                double2 *b,
                                                unsigned long long pitch_b,
                                                unsigned long long length,
                                                unsigned long long N);

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
                                          unsigned long long Nlags,
                                          unsigned long long ipitch,
                                          unsigned long long opitch,
                                          unsigned long long N);

/*! \brief Average power spectrum of input images
    \param d_in     Input complex array
    \param d_out    Output complex array
    \param length   Number of elements in each subarray
    \param pitch    Pitch of input array
    \param Nq       Number of subarrays
*/
__global__ void average_power_spectrum_kernel(double2 *d_in,
                                              double2 *d_out,
                                              unsigned long long length,
                                              unsigned long long pitch,
                                              unsigned long long Nq);

/*! \brief Average over time of Fourier transformed input images
    \param d_in     Input complex array
    \param d_out    Output complex array
    \param length   Number of elements in each subarray
    \param pitch    Pitch of input array
    \param Nq       Number of subarrays
*/
__global__ void average_complex_kernel(double2 *d_in,
                                       double2 *d_out,
                                       unsigned long long length,
                                       unsigned long long pitch,
                                       unsigned long long Nq);

#endif