// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file memchk_gpu.cu
    \brief Definition of utility functions for GPU memory check and optimization
*/

// *** headers ***
#include "memchk_gpu.cuh"
#include "gpu_utils.cuh"
#include "helper_cufft.cuh"
#include "helper_debug.cuh"

#include <cuda.h>
#include <cufft.h>
#include <cuda_runtime.h>

#include <iostream>

// *** code ***

/*!
    Evaluate the device memory pitch for multiple subarrays of size N with 16bytes elements
*/
void cudaGetDevicePitch16B(size_t N,
                           size_t &pitch)
{
    double2 *d_arr;

    gpuErrchk(cudaMallocPitch(&d_arr, &pitch, N * sizeof(double2), 2));

    pitch /= sizeof(double2);

    gpuErrchk(cudaFree(d_arr));
}

/*!
    Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
*/
void cudaGetDevicePitch8B(size_t N,
                          size_t &pitch)
{
    double *d_arr;

    gpuErrchk(cudaMallocPitch(&d_arr, &pitch, N * sizeof(double), 2));

    pitch /= sizeof(double);

    gpuErrchk(cudaFree(d_arr));
}

/*!
    Evaluate the device memory pitch for multiple subarrays of size N with 4bytes elements
*/
void cudaGetDevicePitch4B(size_t N,
                          size_t &pitch)
{
    float *d_arr;

    gpuErrchk(cudaMallocPitch(&d_arr, &pitch, N * sizeof(float), 2));

    pitch /= sizeof(float);

    gpuErrchk(cudaFree(d_arr));
}

/*!
    Evaluate the device memory pitch for multiple subarrays of size N with 2bytes elements
*/
void cudaGetDevicePitch2B(size_t N,
                          size_t &pitch)
{
    int16_t *d_arr;

    gpuErrchk(cudaMallocPitch(&d_arr, &pitch, N * sizeof(int16_t), 2));

    pitch /= sizeof(int16_t);

    gpuErrchk(cudaFree(d_arr));
}

/*!
    Evaluate the device memory pitch for multiple subarrays of size N with 2bytes elements
*/
void cudaGetDevicePitch1B(size_t N,
                          size_t &pitch)
{
    int8_t *d_arr;

    gpuErrchk(cudaMallocPitch(&d_arr, &pitch, N * sizeof(int8_t), 2));

    pitch /= sizeof(int8_t);

    gpuErrchk(cudaFree(d_arr));
}

/*!
    Get device memory pitch (in number of elements)
*/
unsigned long long get_device_pitch(unsigned long long N,
                                    int num_bytes)
{
    size_t pitch;
    switch (num_bytes)
    {
    case 16:
        cudaGetDevicePitch16B(N, pitch);
        break;
    case 8:
        cudaGetDevicePitch8B(N, pitch);
        break;
    case 4:
        cudaGetDevicePitch4B(N, pitch);
        break;
    case 2:
        cudaGetDevicePitch2B(N, pitch);
        break;
    case 1:
        cudaGetDevicePitch1B(N, pitch);
        break;
    default:
        cudaGetDevicePitch8B(N, pitch);
    }

    return (unsigned long long)pitch;
}

/*!
    Optimize fft2 execution parameters based on available gpu memory.

    Writes in the corresponding arguments:
        - the number of iterations for fft2 (frame chunks)
        - the pitch in number of elements for buffer array (real values)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
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
                   unsigned long long &num_fft2)
{
    /*
        Calculations are always performed in double precision.
        However, data is transferred as Scalar (float/double).

        To compute the fft2, we need (values are in bytes):
            - for the buffer (only if input is not Scalar):
                pitch_buff * height * num_fft2 * pixel_Nbytes
            - for the workspace (type: complex double [16 bytes]):
                (nx / 2 + 1) * ny * num_fft2 * 16
            - for the cufft2 internal buffer:
                [determined programmatically...]
            - for the window function (type: Scalar [SCALAR_SIZE bytes]):
                pitch_nx * height * 2 * SCALAR_SIZE
     */

    // Compute the effective number of grid points in x of the rfft2
    unsigned long long _nx = nx / 2ULL + 1ULL;

    // Get the pitch for the buffer array (only if the input is not Scalar)
    pitch_buff = is_input_Scalar ? 0ULL : get_device_pitch(width, pixel_Nbytes);

    // Get the pitch for the rfft2 output complex array
    pitch_nx = get_device_pitch(_nx, 2 * sizeof(Scalar));

    /*
        Start the optimization with the worst case scenario:
        we need to perform as many rfft2 loops as the number of images,
        namely, we transfer 1 image at a time.
     */
    num_fft2 = length;

    // Define auxiliary variables
    unsigned long long mem_required, prev_num_fft2;

    // Optimize
    while (true)
    {
        // Reset required memory value
        mem_required = 0;

        // Compute number of batched transforms
        unsigned long long batch = (length + num_fft2 - 1ULL) / num_fft2;

        // Estimate cufft2 internal work area size
        cufftResult cufft_res;
        unsigned long long mem_fft2 = get_fft2_device_memory_size(nx,
                                                                  ny,
                                                                  batch,
                                                                  pitch_nx,
                                                                  cufft_res);

        // Check if the memory was retrieved successfully
        if (cufft_res == CUFFT_SUCCESS)
        {
            // Add the required memory for cufft2
            mem_required += mem_fft2;

            // Add memory required for the work area
            mem_required += pitch_nx * ny * batch * 16ULL;

            // If input images do not have already the same type as the output,
            // add the required memory for the buffer array
            if (!is_input_Scalar)
            {
                mem_required += pitch_buff * height * batch * (unsigned long long)pixel_Nbytes;
            }

            // If user provided a window function,
            // add memory required for window
            if (is_window)
            {
                mem_required += pitch_nx * ny * 2 * sizeof(Scalar);
            }

            // Check memory and update parameters
            if (free_mem >= mem_required)
            {
                // Estimate the next numer of fft2 loops
                unsigned long long next_num_fft2 = (num_fft2 * mem_required + free_mem - 1ULL) / free_mem;

                // Check if the next number of fft2 loops is the same
                if (next_num_fft2 == prev_num_fft2)
                {
                    break;
                }
                else
                {
                    // Update and repeat
                    prev_num_fft2 = num_fft2;
                    num_fft2 = next_num_fft2;
                }
            }
            else if (num_fft2 == length)
            {
                // In this case, the available memory is less than the required
                // memory and the number of fft2 loops is already the maximum possible.
                // Therefore, we throw an error.
                throw std::runtime_error("Not enough space on GPU for fft2.");
            }
            else
            {
                num_fft2 = prev_num_fft2;
                break;
            }
        }
        else if (num_fft2 == length)
        {
            // The memory was not retrieved successfully
            // and the number of fft2 loops is already the maximum possible.
            // Therefore, we throw an error.
            throw std::runtime_error(
                "Not enough space on GPU for fft2. cufftResult ID: " + cufft_res);
        }
    }
}

/*!
    Optimize structure function "diff" execution parameters based on available gpu memory
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
                                           unsigned long long &pitch_fs)
{
    // Get the available gpu memory
    unsigned long long free_mem = get_free_device_memory();

    // Scale the available memory by 0.9 to leave some free space
    free_mem = (unsigned long long)(0.9 * (double)free_mem);

    // Evaluate parameters for fft2
    optimize_fft2(width,
                  height,
                  length,
                  nx,
                  ny,
                  pixel_Nbytes,
                  is_input_Scalar,
                  is_window,
                  free_mem,
                  pitch_buff,
                  pitch_nx,
                  num_fft2);

    // Evaluate parameters for structure function ("diff" mode)

    // Evaluate parameters for fftshift
}
