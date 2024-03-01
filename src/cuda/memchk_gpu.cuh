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
#include "data_struct.h"

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 16bytes elements
    \param N        Subarray size
    \param pitch    Pitch of the subarray
*/
void cudaGetDevicePitch16B(size_t N,
                           size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        Subarray size
    \param pitch    Pitch of the subarray
*/
void cudaGetDevicePitch8B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        Subarray size
    \param pitch    Pitch of the subarray
*/
void cudaGetDevicePitch4B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        Subarray size
    \param pitch    Pitch of the subarray
*/
void cudaGetDevicePitch2B(size_t N,
                          size_t &pitch);

/*! \brief Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
    \param N        Subarray size
    \param pitch    Pitch of the subarray
*/
void cudaGetDevicePitch1B(size_t N,
                          size_t &pitch);

/*! \brief Get device memory pitch (in number of elements)
    \param N            Subarray size
    \param num_bytes    Element memory size (in bytes)
    \return             Pitch of the subarray
*/
unsigned long long get_device_pitch(unsigned long long N,
                                    int num_bytes);

/*! \brief Optimize FFT2 execution parameters based on available gpu memory
    \param width                    Width of the image
    \param height                   Height of the image
    \param length                   Number of frames
    \param nx                       Number of grid points in x
    \param ny                       Number of grid points in y
    \param nx_half                  Number of grid points of the half-plane representation of the real-to-complex FFT2
    \param input_type_num_bytes     Number of bytes per pixel
    \param is_input_type_scalar     True if image type memory size is same as Scalar
    \param is_window                True if window function is given
    \param free_mem                 Available gpu memory
    \param pitch_buff               Pitch of buffer device array
    \param pitch_nx                 Pitch of fft2 complex output array
    \param num_fft2_loops           Number of fft2 batches
*/
void optimize_fft2(unsigned long long width,
                   unsigned long long height,
                   unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long nx_half,
                   unsigned long long input_type_num_bytes,
                   bool is_input_type_scalar,
                   bool is_window,
                   unsigned long long free_mem,
                   unsigned long long &pitch_buff,
                   unsigned long long &pitch_nx,
                   unsigned long long &num_fft2_loops);

/*! \brief Optimize structure function "diff" execution parameters based on available gpu memory
    \param length           Number of frames
    \param nx               Number of grid points in x
    \param ny               Number of grid points in y
    \param num_lags         Number of lags to be analysed
    \param free_mem         Available gpu memory
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param num_batch_loops  Number of q points batches
*/
void optimize_diff(unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long num_lags,
                   unsigned long long free_mem,
                   unsigned long long &pitch_q,
                   unsigned long long &pitch_t,
                   unsigned long long &num_batch_loops);

/*! \brief Optimize structure function "fft" execution parameters based on available gpu memory
    \param length           Number of frames
    \param nx               Number of grid points in x
    \param ny               Number of grid points in y
    \param nt               Number of grid points in t
    \param num_lags         Number of lags to be analysed
    \param free_mem         Available gpu memory
    \param pitch_q          Pitch of device array (q-pitch)
    \param pitch_t          Pitch of device array (t-pitch)
    \param pitch_nt         Pitch of device array (nt-pitch)
    \param num_batch_loops  Number of q points batches
*/
void optimize_fft(unsigned long long length,
                  unsigned long long nx,
                  unsigned long long ny,
                  unsigned long long nt,
                  unsigned long long num_lags,
                  unsigned long long free_mem,
                  unsigned long long &pitch_q,
                  unsigned long long &pitch_t,
                  unsigned long long &pitch_nt,
                  unsigned long long &num_batch_loops);

/*! \brief Optimize fftshift execution parameters based on available gpu memory
    \param nx                   Number of grid points in x
    \param ny                   Number of grid points in y
    \param num_lags             Number of lags to be analysed
    \param free_mem             Available gpu memory
    \param pitch_fs             Pitch of device array for shift operation
    \param num_fftshift_loops   Number of fftshift batches
*/
void optimize_fftshift(unsigned long long nx,
                       unsigned long long ny,
                       unsigned long long num_lags,
                       unsigned long long free_mem,
                       unsigned long long &pitch_fs,
                       unsigned long long &num_fftshift_loops);

/*! \brief Optimize "diff" execution parameters based on available gpu memory
    \param img_data     Structure holding the image sequence parameters
    \param sf_data      Structure holding the structure function parameters
    \param exec_params  Structure holding the execution parameters
    \param pitch_data   Structure holding the memory pitch parameters
*/
void check_and_optimize_device_memory_diff(ImageData &img_data,
                                           StructureFunctionData &sf_data,
                                           ExecutionParameters &exec_params,
                                           PitchData &pitch_data);

/*! \brief Optimize "fft" execution parameters based on available gpu memory
    \param nt           Number of grid points in t
    \param img_data     Structure holding the image sequence parameters
    \param sf_data      Structure holding the structure function parameters
    \param exec_params  Structure holding the execution parameters
    \param pitch_data   Structure holding the memory pitch parameters
*/
void check_and_optimize_device_memory_fft(unsigned long long nt,
                                          ImageData &img_data,
                                          StructureFunctionData &sf_data,
                                          ExecutionParameters &exec_params,
                                          PitchData &pitch_data);

#endif // __MEMCHK_GPU_CUH__
