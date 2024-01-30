// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __HELPER_PREFIX_SUM_CUH__
#define __HELPER_PREFIX_SUM_CUH__

/*! \file helper_prefix_sum.cuh
    \brief Declaration of helper functions for cumulative sum (prefix sum) on GPU
*/

// *** headers ***

// *** code ***

/*! \brief Compute next power of two larger or equal to n
    \param n    Target
 */
unsigned long long nextPowerOfTwo(unsigned long long n);

/*! \brief Scan multiple large arrays on the GPU
    \param output   Output array
    \param input    Input array
    \param length   Number of elements in each subarray
    \param dist     Distance between the first element of two consecutive subarrays
    \param N        Number of subarrays
 */
void scanManyLargeArrays(double *output,
                         double *input,
                         unsigned long long length,
                         unsigned long long dist,
                         unsigned long long N);

/*! \brief Scan multiple large even arrays on the GPU
    \param output   Output array
    \param input    Input array
    \param length   Number of elements in each subarray
    \param dist     Distance between the first element of two consecutive subarrays
    \param Nx       Number of even blocks per row
    \param N        Number of subarrays
 */
void scanManyLargeEvenArrays(double *output,
                             double *input,
                             unsigned long long length,
                             unsigned long long dist,
                             unsigned long long Nx,
                             unsigned long long N);

/*! \brief Scan multiple small arrays on the GPU
    \param output   Output array
    \param input    Input array
    \param length   Number of elements in each subarray
    \param dist     Distance between the first element of two consecutive subarrays
    \param N        Number of subarrays
 */
void scanManySmallArrays(double *output,
                         double *input,
                         unsigned long long length,
                         unsigned long long dist,
                         unsigned long long N);

/*! \brief Compute cumulative sum on multiple subarrays on the GPU
    \param output   Output array
    \param input    Input array
    \param length   Number of elements in each subarray
    \param dist     Distance between the first element of two consecutive subarrays
    \param N        Number of subarrays
 */
void scan_wrap(double *output,
               double *input,
               unsigned long long length,
               unsigned long long dist,
               unsigned long long N);

/*! \brief Compute cumulative sum on multiple even portions of the subarrays on the GPU
    \param output   Output array
    \param input    Input array
    \param dist     Distance between the first element of two consecutive subarrays
    \param Nx       Number of blocks per subarray
    \param NxNy     Total number of blocks
    \param n        Number of elements in each block
    \param sums     Intermediate array for sums
 */
__global__ void prescan_many_even_kernel(double *output,
                                         double *input,
                                         unsigned long long dist,
                                         unsigned long long Nx,
                                         unsigned long long NxNy,
                                         unsigned long long n,
                                         double *sums);

/*! \brief Compute cumulative sum on multiple arbitrary (small) portions of the subarrays on the GPU
    \param output       Output array
    \param input        Input array
    \param dist         Distance between the first element of two consecutive subarrays
    \param N            Total number of blocks
    \param n            Number of elements in each block
    \param powerOfTwo   Next power of two >= n
 */
__global__ void prescan_many_arbitrary_kernel(double *output,
                                              double *input,
                                              unsigned long long dist,
                                              unsigned long long N,
                                              unsigned long long n,
                                              unsigned long long powerOfTwo);

/*! \brief Add block dependent constant value to every element in the block
    \param output       Output array
    \param dist         Distance between the first element of two consecutive subarrays
    \param Nx           Number of blocks per subarray
    \param NxNy         Total number of blocks
    \param n            Number of elements in each block
    \param a            Constant
 */
__global__ void add_many_kernel(double *output,
                                unsigned long long dist,
                                unsigned long long Nx,
                                unsigned long long NxNy,
                                unsigned long long n,
                                double *a);

/*! \brief Add two block dependent constant values to every element in the block
    \param output       Output array
    \param dist         Distance between the first element of two consecutive subarrays
    \param Nx           Number of blocks per subarray
    \param NxNy         Total number of blocks
    \param n            Number of elements in each block
    \param a1           Constant 1
    \param a2           Constant 2
 */
__global__ void add_many_kernel(double *output,
                                unsigned long long dist,
                                unsigned long long Nx,
                                unsigned long long NxNy,
                                unsigned long long n,
                                double *a1,
                                double *a2);

/*! \brief Copy element from input to output every dist elements
    \param output       Output array
    \param input        Input array
    \param dist         Distance between two consecutive elements to be copied
    \param N            Total number of elements
 */
__global__ void copy_every_kernel(double *output,
                                  double *input,
                                  unsigned long long dist,
                                  unsigned long long N);

#endif // __HELPER_PREFIX_SUM_CUH__
