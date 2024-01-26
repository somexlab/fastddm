// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

#ifndef __MEMCHK_GPU_CUH__
#define __MEMCHK_GPU_CUH__

/*! \file memchk_gpu.cuh
    \brief Declaration of utility functions for GPU memory check and optimization
*/

// *** headers ***
#include "../python_defs.h"

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

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

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        subarray size
    \param pitch    pitch of the subarray
*/
void cudaGetDevicePitch4B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        subarray size
    \param pitch    pitch of the subarray
*/
void cudaGetDevicePitch2B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        subarray size
    \param pitch    pitch of the subarray
*/
void cudaGetDevicePitch1B(size_t N,
                          size_t &pitch);

/*! \brief Get device memory pitch (in number of elements)
    \param N            subarray size
    \param num_bytes    element memory size (in bytes)
    \return             pitch of the subarray
*/
unsigned long long get_device_pitch(unsigned long long N,
                                    int num_bytes);

/*! \brief Optimize fft2 execution parameters based on available gpu memory
    \param width            Width of the image
    \param height           Height of the image
    \param length           Number of frames
    \param nx               Number of grid points in x
    \param ny               Number of grid points in y
    \param pixel_Nbytes     Number of bytes per pixel
    \param is_input_Scalar  True if image type memory size is same as Scalar
    \param is_window        True if window function is given
    \param free_mem         Available gpu memory
    \param pitch_buff       Pitch of buffer device array
    \param pitch_nx         Pitch of fft2 complex output array
    \param num_fft2         Number of fft2 batches
*/
void optimize_fft2(unsigned long long width,
                   unsigned long long height,
                   unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long pixel_Nbytes,
                   bool is_input_Scalar,
                   bool is_window,
                   unsigned long long free_mem,
                   unsigned long long &pitch_buff,
                   unsigned long long &pitch_nx,
                   unsigned long long &num_fft2);

/*! \brief Optimize structure function "diff" execution parameters based on available gpu memory
    \param length           Number of frames
    \param nx               Number of grid points in x
    \param ny               Number of grid points in y
    \param num_lags         Number of lags to be analysed
    \param free_mem         Available gpu memory
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param num_chunks       Number of q points chunks
*/
void optimize_diff(unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long num_lags,
                   unsigned long long free_mem,
                   unsigned long long &pitch_q,
                   unsigned long long &pitch_t,
                   unsigned long long &num_chunks);

/*! \brief Optimize fftshift execution parameters based on available gpu memory
    \param nx               Number of grid points in x
    \param ny               Number of grid points in y
    \param num_lags         Number of lags to be analysed
    \param free_mem         Available gpu memory
    \param pitch_fs         Pitch of device array for shift operation
    \param num_shift        Number of shift chunks
*/
void optimize_fftshift(unsigned long long nx,
                       unsigned long long ny,
                       unsigned long long num_lags,
                       unsigned long long free_mem,
                       unsigned long long &pitch_fs,
                       unsigned long long &num_shift);

/*! \brief Optimize "diff" execution parameters based on available gpu memory
    \param width            Width of the image
    \param height           Height of the image
    \param length           Number of frames
    \param num_lags         Number of lags analysed
    \param nx               Number of fft nodes, x direction
    \param ny               Number of fft nodes, y direction
    \param pixel_Nbytes     Number of bytes per pixel
    \param is_input_Scalar  True if image type memory size is same as Scalar
    \param is_window        True if window function is given
    \param num_fft2         Number of fft2 batches
    \param num_chunks       Number of q points chunks
    \param num_shift        Number of shift chunks
    \param pitch_buff       Pitch of buffer device array
    \param pitch_nx         Pitch of device fft2 output array (nx-pitch)
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param pitch_fs         Pitch of device array for shift operation
*/
void check_and_optimize_device_memory_diff(unsigned long long width,
                                           unsigned long long height,
                                           unsigned long long length,
                                           unsigned long long num_lags,
                                           unsigned long long nx,
                                           unsigned long long ny,
                                           int pixel_Nbytes,
                                           bool is_input_Scalar,
                                           bool is_window,
                                           unsigned long long &num_fft2,
                                           unsigned long long &num_chunks,
                                           unsigned long long &num_shift,
                                           unsigned long long &pitch_buff,
                                           unsigned long long &pitch_nx,
                                           unsigned long long &pitch_q,
                                           unsigned long long &pitch_t,
                                           unsigned long long &pitch_fs);

#endif // __MEMCHK_GPU_CUH__
