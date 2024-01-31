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
                   unsigned long long nx_half,
                   unsigned long long input_type_num_bytes,
                   bool is_input_type_scalar,
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

    // Get the pitch for the buffer array (only if the input is not Scalar)
    pitch_buff = is_input_type_scalar ? 0ULL : get_device_pitch(width, input_type_num_bytes);

    // Get the pitch for the rfft2 output complex array
    pitch_nx = get_device_pitch(nx_half, 2 * sizeof(Scalar));

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

        // Check if the memory was retrieved correctly
        if (cufft_res == CUFFT_SUCCESS)
        {
            // Add the memory required for cufft2
            mem_required += mem_fft2;

            // Add memory required for the work area
            mem_required += pitch_nx * ny * batch * 16ULL;

            // If input images do not have already the same type as the output,
            // add the required memory for the buffer array
            if (!is_input_type_scalar)
            {
                mem_required += pitch_buff * height * batch * input_type_num_bytes;
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
                // We already found a good number of fft2 loops previously,
                // so we go back to the previous value and stop the optimization
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

    Writes in the corresponding arguments:
        - the pitch in number of elements for workspace (pitch_q, complex double)
        - the pitch in number of elements for workspace (pitch_t, complex double)
        - the number of iterations for structure function (num_batch_loops, q-vector batches)

    Throws a runtime_error if the memory is not sufficient to perform the calculations.
*/
void optimize_diff(unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long num_lags,
                   unsigned long long free_mem,
                   unsigned long long &pitch_q,
                   unsigned long long &pitch_t,
                   unsigned long long &num_batch_loops)
{
    /*
        Calculations are always performed in double precision.
        However, data is transferred as Scalar (float/double).

        To compute the structure function in "diff" mode, we need (values are in bytes):
            - for the lags helper array (type: unsigned int [4 bytes])
                num_lags * 4
            - for the workspace1 and workspace2 arrays (type: complex double [16 bytes])
                2 * max(pitch_q * length, batch_size * pitch_t) * 16
            - for the power_spec and var helper arrays (type: double2 [16 bytes])
                2 * batch_size * 16
    */

    // Get the pitch for the workspace array (time pitch, complex double)
    pitch_t = get_device_pitch(length, 16);

    /*
        Start the optimization with the worst case scenario:
        we need to perform as many loops as the number of q vectors
    */
    num_batch_loops = (nx / 2ULL + 1ULL) * ny;

    // Define auxiliary variables
    unsigned long long mem_required, prev_num_batch_loops;

    // Optimize
    while (true)
    {
        // Reset required memory value
        mem_required = 0;

        // Compute the number of q-vectors per batch
        unsigned long long batch_size = ((nx / 2ULL + 1ULL) * ny + num_batch_loops - 1ULL) / num_batch_loops;

        // Get the pitch for the workspace array (q pitch, complex Scalar)
        pitch_q = get_device_pitch(batch_size, 2 * sizeof(Scalar));

        // Add the required memory for the lags helper array
        mem_required += num_lags * 4ULL;

        // Add the required memory for the two workspace arrays
        mem_required += 2ULL * max(pitch_q * length, batch_size * pitch_t) * 16ULL;

        // Add the memory required for the power_spec and var helper arrays
        mem_required += 2ULL * batch_size * 16ULL;

        // Check memory and update parameters
        if (free_mem >= mem_required)
        {
            // Estimate the next number of batches
            unsigned long long next_num_batch_loops = (num_batch_loops * mem_required + free_mem - 1ULL) / free_mem;

            // Check if the next number of batches is the same
            if (next_num_batch_loops == prev_num_batch_loops)
            {
                break;
            }
            else
            {
                prev_num_batch_loops = num_batch_loops;
                num_batch_loops = next_num_batch_loops;
            }
        }
        else if (num_batch_loops == (nx / 2ULL + 1ULL) * ny)
        {
            // In this case, the available memory is less than the required
            // memory and the number of batches is already the maximum possible.
            // Therefore, we throw an error.
            throw std::runtime_error("Not enough space on GPU for structure function calculation.");
        }
        else
        {
            // We already found a good number of batches previously,
            // so we go back to the previous value and stop the optimization
            num_batch_loops = prev_num_batch_loops;

            // We need to update the pitch for the workspace array (q pitch)
            batch_size = ((nx / 2ULL + 1ULL) * ny + num_batch_loops - 1ULL) / num_batch_loops;
            pitch_q = get_device_pitch(batch_size, 2 * sizeof(Scalar));

            break;
        }
    }
}

/*!
    Optimize structure function "fft" execution parameters based on available gpu memory

    Writes in the corresponding arguments:
        - the pitch in number of elements for workspace (pitch_q, complex double)
        - the pitch in number of elements for workspace (pitch_t, complex double)
        - the pitch in number of elements for workspace (pitch_nt, complex double)
        - the number of iterations for structure function (num_batch_loops, q-vector batches)

    Throws a runtime_error if the memory is not sufficient to perform the calculations.
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
                  unsigned long long &num_batch_loops)
{
    /*
        Calculations are always performed in double precision.
        However, data is transferred as Scalar (float/double).

        To compute the structure function in "diff" mode, we need (values are in bytes):
            - for the lags helper array (type: unsigned int [4 bytes])
                num_lags * 4
            - for the workspace1 array (type: complex double [16 bytes])
                batch_size * pitch_nt * 16
            - for the workspace2 array (type: complex double [16 bytes])
                max(pitch_q * length, batch_size * pitch_t) * 16
            - for the cufft internal buffer:
                [determined programmatically...]
            - for the prefix sum buffer (type: double [8 bytes])
                (batch_size + max((2 * length / 1024) * batch_size, batch_size)) * 8
            - for the power_spec and var helper arrays (type: double2 [16 bytes])
                2 * batch_size * 16
    */

    // Get the pitch for the workspace array (time pitch, complex double)
    pitch_t = get_device_pitch(length, 16);

    // Get the pitch for the workspace array (fft pitch, complex double)
    pitch_nt = get_device_pitch(nt, 16);

    /*
        Start the optimization with the worst case scenario:
        we need to perform as many loops as the number of q vectors
    */
    num_batch_loops = (nx / 2ULL + 1ULL) * ny;

    // Define auxiliary variables
    unsigned long long mem_required, prev_num_batch_loops;

    // Optimize
    while (true)
    {
        // Reset required memory value
        mem_required = 0;

        // Compute the number of q-vectors per batch
        unsigned long long batch_size = ((nx / 2ULL + 1ULL) * ny + num_batch_loops - 1ULL) / num_batch_loops;

        // Estimate cufft internal work area size
        cufftResult cufft_res;
        unsigned long long mem_fft = get_fft_device_memory_size(nt,
                                                                batch_size,
                                                                pitch_nt,
                                                                cufft_res);

        // Check if the memory was retrieved correctly
        if (cufft_res == CUFFT_SUCCESS)
        {
            // Add the memory required for the cufft
            mem_required += mem_fft;

            // Get the pitch for the workspace array (q pitch, complex Scalar)
            pitch_q = get_device_pitch(batch_size, 2 * sizeof(Scalar));

            // Add the required memory for the lags helper array
            mem_required += num_lags * 4ULL;

            // Add the required memory for the workspace1 array
            mem_required += batch_size * pitch_nt * 16ULL;

            // Add the required memory for the workspace2 array
            mem_required += max(pitch_q * length, batch_size * pitch_t) * 16ULL;

            // Add the required memory for the prefix sum buffer
            mem_required += 2ULL * length < 1024ULL ? 2ULL * batch_size * 8ULL : batch_size * (1ULL + 2ULL * ((2ULL * length) / 1024ULL)) * 8ULL;

            // Add the required memory for the power_spec and var helper arrays
            mem_required += 2ULL * batch_size * 16ULL;

            // Check memory and update parameters
            if (free_mem >= mem_required)
            {
                // Estimate the next number of batches
                unsigned long long next_num_batch_loops = (num_batch_loops * mem_required + free_mem - 1ULL) / free_mem;

                // Check if the next number of batches is the same
                if (next_num_batch_loops == prev_num_batch_loops)
                {
                    break;
                }
                else
                {
                    prev_num_batch_loops = num_batch_loops;
                    num_batch_loops = next_num_batch_loops;
                }
            }
            else if (num_batch_loops == (nx / 2ULL + 1ULL) * ny)
            {
                // In this case, the available memory is less than the required
                // memory and the number of batches is already the maximum possible.
                // Therefore, we throw an error.
                throw std::runtime_error("Not enough space on GPU for structure function calculation.");
            }
            else
            {
                // We already found a good number of batches previously,
                // so we go back to the previous value and stop the optimization
                num_batch_loops = prev_num_batch_loops;

                // We need to update the pitch for the workspace array (q pitch)
                batch_size = ((nx / 2ULL + 1ULL) * ny + num_batch_loops - 1ULL) / num_batch_loops;
                pitch_q = get_device_pitch(batch_size, 2 * sizeof(Scalar));

                break;
            }
        }
        else if (num_batch_loops == (nx / 2ULL + 1ULL) * ny)
        {
            throw std::runtime_error(
                "Not enough space on GPU for structure function calculation with fft. cufftResult #: " + cufft_res);
        }
    }
}

/*!
    Optimize fftshift execution parameters based on available gpu memory

    Writes in the corresponding arguments:
        - the number of iterations for fftshift (num_fftshift_loops)
        - the pitch in number of elements for shift workspace (pitch_fs, complex Scalar)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
*/
void optimize_fftshift(unsigned long long nx,
                       unsigned long long ny,
                       unsigned long long num_lags,
                       unsigned long long free_mem,
                       unsigned long long &pitch_fs,
                       unsigned long long &num_fftshift_loops)
{
    /*
        Calculations are performed in single or double precision, based on the output.

        To perform the fftshift, we need (values are in bytes):
            - for the workspace1 (type: complex Scalar [2 * sizeof(Scalar) bytes])
                pitch_fs * ny * batch_size * 2 * sizeof(Scalar)
            - for the workspace2 (type: Scalar [sizeof(Scalar) bytes])
                pitch_fs * ny * batch_size * sizeof(Scalar)
    */

    // Get the pitch for the workspace arrays (shift pitch, complex Scalar)
    pitch_fs = get_device_pitch((nx / 2ULL + 1ULL), 2 * sizeof(Scalar));

    /*
        Start the optimization with the worst case scenario:
        we need to perform as many fftshift loops as the number of lags
    */
    num_fftshift_loops = num_lags;

    // Define auxiliary variables
    unsigned long long mem_required, prev_num_fftshift_loops;

    // Optimize
    while (true)
    {
        // Reset required memory value
        mem_required = 0;

        // Compute the number of batched fftshifts
        unsigned long long batch_size = (num_lags + num_fftshift_loops - 1ULL) / num_fftshift_loops;

        // Add the required memory for the two workspace arrays
        mem_required += pitch_fs * ny * batch_size * 2ULL * sizeof(Scalar);
        mem_required += pitch_fs * ny * batch_size * sizeof(Scalar);

        // Check memory and update parameters
        if (free_mem >= mem_required)
        {
            // Estimate the next number of fftshift loops
            unsigned long long next_num_fftshift_loops = (num_fftshift_loops * mem_required + free_mem - 1ULL) / free_mem;

            // Check if the next number of fftshift loops is the same
            if (next_num_fftshift_loops == prev_num_fftshift_loops)
            {
                break;
            }
            else
            {
                prev_num_fftshift_loops = num_fftshift_loops;
                num_fftshift_loops = next_num_fftshift_loops;
            }
        }
        else if (num_fftshift_loops == num_lags)
        {
            // In this case, the available memory is less than the required
            // memory and the number of fftshift loops is already the maximum possible.
            // Therefore, we throw an error.
            throw std::runtime_error("Not enough space on GPU for fftshift.");
        }
        else
        {
            // We already found a good number of fftshift loops previously,
            // so we go back to the previous value and stop the optimization
            num_fftshift_loops = prev_num_fftshift_loops;

            break;
        }
    }
}

/*!
    Optimize "diff" execution parameters based on available gpu memory
*/
void check_and_optimize_device_memory_diff(ImageData &img_data,
                                           StructureFunctionData &sf_data,
                                           ExecutionParameters &exec_params,
                                           PitchData &pitch_data)
{
    // Get the available gpu memory
    unsigned long long free_mem = get_free_device_memory();

    // Scale the available memory by 0.9 to leave some free space
    free_mem = (unsigned long long)(0.9 * (double)free_mem);

    // Evaluate parameters for fft2
    optimize_fft2(img_data.width,
                  img_data.height,
                  img_data.length,
                  sf_data.nx,
                  sf_data.ny,
                  sf_data.nx_half,
                  img_data.input_type_num_bytes,
                  img_data.is_input_type_scalar,
                  sf_data.is_window,
                  free_mem,
                  pitch_data.p_buffer,
                  pitch_data.p_nx,
                  exec_params.num_fft2_loops);

    // Evaluate parameters for structure function ("diff" mode)
    optimize_diff(img_data.length,
                  sf_data.nx,
                  sf_data.ny,
                  sf_data.num_lags,
                  free_mem,
                  pitch_data.p_q,
                  pitch_data.p_t,
                  exec_params.num_batch_loops);

    // Evaluate parameters for fftshift
    optimize_fftshift(sf_data.nx,
                      sf_data.ny,
                      sf_data.num_lags,
                      free_mem,
                      pitch_data.p_fftshift,
                      exec_params.num_fftshift_loops);
}

/*!
    Optimize "fft" execution parameters based on available gpu memory
*/
void check_and_optimize_device_memory_fft(unsigned long long nt,
                                          ImageData &img_data,
                                          StructureFunctionData &sf_data,
                                          ExecutionParameters &exec_params,
                                          PitchData &pitch_data)
{
    // Get the available gpu memory
    unsigned long long free_mem = get_free_device_memory();

    // Scale the available memory by 0.9 to leave some free space
    free_mem = (unsigned long long)(0.9 * (double)free_mem);

    // Set nt in the execution parameters
    exec_params.nt = nt;

    // Evaluate parameters for fft2
    optimize_fft2(img_data.width,
                  img_data.height,
                  img_data.length,
                  sf_data.nx,
                  sf_data.ny,
                  sf_data.nx_half,
                  img_data.input_type_num_bytes,
                  img_data.is_input_type_scalar,
                  sf_data.is_window,
                  free_mem,
                  pitch_data.p_buffer,
                  pitch_data.p_nx,
                  exec_params.num_fft2_loops);

    // Evaluate parameters for structure function ("fft" mode)
    optimize_fft(img_data.length,
                 sf_data.nx,
                 sf_data.ny,
                 exec_params.nt,
                 sf_data.num_lags,
                 free_mem,
                 pitch_data.p_q,
                 pitch_data.p_t,
                 pitch_data.p_nt,
                 exec_params.num_batch_loops);

    // Evaluate parameters for fftshift
    optimize_fftshift(sf_data.nx,
                      sf_data.ny,
                      sf_data.num_lags,
                      free_mem,
                      pitch_data.p_fftshift,
                      exec_params.num_fftshift_loops);
}
