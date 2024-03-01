// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file ddm_cuda.cuh
    \brief Declaration of core Differential Dynamic Microscopy functions for GPU
*/

// *** headers ***
#include "ddm_cuda.cuh"
#include "gpu_utils.cuh"
#include "helper_cufft.cuh"
#include "helper_ddm_cuda.cuh"
#include "helper_prefix_sum.cuh"

#include "helper_debug.cuh"

#include <cufft.h>
#include <cuda_runtime.h>

#include <cstdint>

#define CUFFTCOMPLEX cufftDoubleComplex

#ifndef SINGLE_PRECISION
typedef double2 Scalar2;
#else
typedef float2 Scalar2;
#endif

// *** code ***
const unsigned long long TILE_DIM = 32;  // leave this unchanged!! (tile dimension for matrix transpose)
const unsigned long long BLOCK_ROWS = 8; // leave this unchanged!! (block rows for matrix transpose)

/*!
    Transfer images on GPU and compute fft2
 */
template <typename T>
void compute_fft2(const T *h_in,
                  Scalar *h_out,
                  const Scalar *h_window,
                  ImageData &img_data,
                  StructureFunctionData &sf_data,
                  ExecutionParameters &exec_params,
                  PitchData &pitch_data)
{
    // Compute the number of batched fft2
    unsigned long long batch = (img_data.length - 1ULL) / exec_params.num_fft2_loops + 1ULL;
    // Compute the fft2 normalization factor
    double norm_fact = 1.0 / sqrt((double)(sf_data.nx * sf_data.ny));

    // *** Allocate device arrays
    // Allocate workspace
    double *d_workspace;
    unsigned long long workspace_size = 2ULL * pitch_data.p_nx * sf_data.ny * batch * sizeof(double);
    gpuErrchk(cudaMalloc(&d_workspace, workspace_size));
    // Allocate buffer space (only if input type is not double already)
    T *d_buff;
    if (!std::is_same<T, double>::value)
    {
        unsigned long long buffer_size = pitch_data.p_buffer * img_data.height * batch * sizeof(T);
        gpuErrchk(cudaMalloc(&d_buff, buffer_size));
    }
    // Allocate window space (only if window is given by the user)
    Scalar *d_window;
    if (sf_data.is_window)
    {
        unsigned long long window_size = 2ULL * pitch_data.p_nx * sf_data.ny * sizeof(double);
        gpuErrchk(cudaMalloc(&d_window, window_size));
    }

    // *** Create the fft2 plan
    cufftHandle fft2_plan = create_fft2_plan(sf_data.nx,
                                             sf_data.ny,
                                             batch,
                                             pitch_data.p_nx);

    // *** Estimate efficient execution configuration
    // Get device id
    int device_id = get_device();
    // Get number of streaming multiprocessors
    int numSMs;
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id));
    // Get maximum grid size
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));
    // Parameters for copy_convert_kernel
    int blockSize_copy; // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_copy, copy_convert_kernel<T>, 0, 0));
    dim3 gridSize_copy(min(1ULL * maxGridSizeX, batch), min(1ULL * maxGridSizeY, img_data.height), 1ULL);
    // Parameters for apply_window_kernel
    int blockSize_win;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_win, apply_window_kernel, 0, 0));
    dim3 gridSize_win(min(1ULL * maxGridSizeX, batch), min(1ULL * maxGridSizeY, img_data.height), 1ULL);
    // Parameters for scale_array_kernel
    int blockSize_scale; // The launch configurator returned block size
    int gridSize_scale;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_scale, scale_array_kernel, 0, 0));
    // Round up according to array size
    gridSize_scale = min((sf_data.ny * batch + blockSize_scale - 1ULL) / blockSize_scale, 32ULL * numSMs);

#ifdef SINGLE_PRECISION
    // in place conversion kernel
    int smem_size = sf_data.nx_half * sizeof(float2);
#endif

    // *** Compute fft2
    for (unsigned long long ii = 0; ii < exec_params.num_fft2_loops; ii++)
    {
        // *** Reset workspace array to 0 (ensures zero padding)
        gpuErrchk(cudaMemset(d_workspace, 0.0, workspace_size));

        // *** Transfer images to GPU
        // If input is double, transfer memory directly with pitch
        if (std::is_same<T, double>::value)
        {
            // Compute the number of images to copy
            unsigned long long num_imgs_copy = batch;
            if ((ii + 1) * batch > img_data.length)
            {
                num_imgs_copy = img_data.length - ii * batch;
            }

            // Copy with cudaMemcpy3D
            cudaMemcpy3DParms params = {0};
            params.srcArray = NULL;
            params.srcPos = make_cudaPos(0, 0, ii * batch);
            params.srcPtr = make_cudaPitchedPtr((double *)h_in,
                                                img_data.width * sizeof(double),
                                                img_data.width,
                                                img_data.height);
            params.dstArray = NULL;
            params.dstPos = make_cudaPos(0, 0, 0);
            params.dstPtr = make_cudaPitchedPtr(d_workspace,
                                                2ULL * pitch_data.p_nx * sizeof(double),
                                                2ULL * pitch_data.p_nx,
                                                sf_data.ny);
            params.extent = make_cudaExtent(img_data.width * sizeof(double),
                                            img_data.height,
                                            num_imgs_copy);
            params.kind = cudaMemcpyHostToDevice;

            gpuErrchk(cudaMemcpy3D(&params));
        }
        else // If input is not double, transfer memory to buffer and convert to double to workspace
        {
            // Reset buffer array to 0
            gpuErrchk(cudaMemset(d_buff, (T)0, pitch_data.p_buffer * img_data.height * batch * sizeof(T)));

            // Compute offset index (starting index of the current batch)
            unsigned long long offset = ii * img_data.width * img_data.height * batch;
            // Compute number of rows to copy
            unsigned long long num_rows_copy = img_data.height * batch;
            if ((ii + 1) * batch > img_data.length)
            {
                num_rows_copy = img_data.height * (img_data.length - ii * batch);
            }
            // Copy values to buffer
            gpuErrchk(cudaMemcpy2D(d_buff,
                                   pitch_data.p_buffer * sizeof(T),
                                   h_in + offset,
                                   img_data.width * sizeof(T),
                                   img_data.width * sizeof(T),
                                   num_rows_copy,
                                   cudaMemcpyHostToDevice));

            // Convert values of buffer into workspace
            copy_convert_kernel<<<gridSize_copy, blockSize_copy>>>(d_buff,
                                                                   d_workspace,
                                                                   img_data.width,
                                                                   img_data.height,
                                                                   batch,
                                                                   pitch_data.p_buffer,
                                                                   pitch_data.p_buffer * img_data.height,
                                                                   2ULL * pitch_data.p_nx,
                                                                   2ULL * pitch_data.p_nx * sf_data.ny);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        // *** Apply the window function
        if (sf_data.is_window)
        {
            // copy window function to device
            gpuErrchk(cudaMemcpy2D(d_window,
                                   2ULL * pitch_data.p_nx * sizeof(Scalar),
                                   h_window,
                                   img_data.width * sizeof(Scalar),
                                   img_data.width * sizeof(Scalar),
                                   img_data.height,
                                   cudaMemcpyHostToDevice));
            // apply window
            apply_window_kernel<<<gridSize_win, blockSize_win>>>(d_workspace,
                                                                 d_window,
                                                                 img_data.width,
                                                                 img_data.height,
                                                                 batch,
                                                                 2 * pitch_data.p_nx,
                                                                 2 * pitch_data.p_nx * sf_data.ny,
                                                                 2 * pitch_data.p_nx);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        // ***Execute fft2 plan
        cufftSafeCall(cufftExecD2Z(fft2_plan, d_workspace, (CUFFTCOMPLEX *)d_workspace));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Normalize fft2
        // Starting index
        unsigned long long start = ii * sf_data.ny * batch;
        // Final index (if exceeds array size, truncate)
        unsigned long long end = (ii + 1) * batch > img_data.length ? img_data.length * sf_data.ny : (ii + 1) * sf_data.ny * batch;
        // scale array
        scale_array_kernel<<<gridSize_scale, blockSize_scale>>>((double2 *)d_workspace,
                                                                pitch_data.p_nx,
                                                                sf_data.nx_half,
                                                                norm_fact,
                                                                end - start);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

#ifdef SINGLE_PRECISION
        // ***Convert workspace array in place from double2 to float2
        // The array has the same pitch (in bytes), but the elements are contiguous in memory
        // (within one row)
        double2float_inplace_kernel<<<gridSize_scale, blockSize_scale, smem_size>>>((double2 *)d_workspace,
                                                                                    (float2 *)d_workspace,
                                                                                    pitch_data.p_nx,
                                                                                    sf_data.nx_half,
                                                                                    end - start);
#endif

        // ***Copy values back to host
        gpuErrchk(cudaMemcpy2D((Scalar2 *)h_out + sf_data.nx_half * start,
                               sf_data.nx_half * sizeof(Scalar2),
                               (Scalar2 *)d_workspace,
                               pitch_data.p_nx * sizeof(double2),
                               sf_data.nx_half * sizeof(Scalar2),
                               end - start,
                               cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace));
    if (!std::is_same<T, double>::value)
    {
        gpuErrchk(cudaFree(d_buff));
    }
    if (sf_data.is_window)
    {
        gpuErrchk(cudaFree(d_window));
    }
    cufftSafeCall(cufftDestroy(fft2_plan));
}

template void compute_fft2<uint8_t>(const uint8_t *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<int16_t>(const int16_t *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<uint16_t>(const uint16_t *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<int32_t>(const int32_t *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<uint32_t>(const uint32_t *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<int64_t>(const int64_t *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<uint64_t>(const uint64_t *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<float>(const float *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);
template void compute_fft2<double>(const double *h_in, Scalar *h_out, const Scalar *h_window, ImageData &img_data, StructureFunctionData &sf_data, ExecutionParameters &exec_params, PitchData &pitch_data);

/*!
    Compute structure function using differences on the GPU
 */
void structure_function_diff(Scalar *h_in,
                             vector<unsigned int> lags,
                             ImageData &img_data,
                             StructureFunctionData &sf_data,
                             ExecutionParameters &exec_params,
                             PitchData &pitch_data)
{
    // Compute the number of q points in each batch
    unsigned long long batch_size = (sf_data.nx_half * sf_data.ny - 1ULL) / exec_params.num_batch_loops + 1ULL;

    // *** Allocate device arrays
    // Allocate workspaces
    double *d_workspace1, *d_workspace2;
    unsigned long long workspace_size = max(pitch_data.p_q * img_data.length, batch_size * pitch_data.p_t) * 2ULL * sizeof(double);
    gpuErrchk(cudaMalloc(&d_workspace1, workspace_size));
    gpuErrchk(cudaMalloc(&d_workspace2, workspace_size));
    // Allocate helper arrays
    unsigned int *d_lags;
    double2 *d_power_spec, *d_var;
    gpuErrchk(cudaMalloc(&d_lags, sf_data.num_lags * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_power_spec, batch_size * sizeof(double2)));
    gpuErrchk(cudaMalloc(&d_var, batch_size * sizeof(double2)));

    // Copy lags to device
    gpuErrchk(cudaMemcpy(d_lags, lags.data(), sf_data.num_lags * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // *** Estimate efficient execution configuration
    // Get device id
    int device_id = get_device();
    // Get number of streaming multiprocessors
    int numSMs;
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id));
    // Parameters for float2double_kernel
    int minGridSize;   // The minimum grid size needed to achieve the
                       // maximum occupancy for a full device launch
    int blockSize_f2d; // The launch configurator returned block size
    int gridSize_f2d;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_f2d, float2double_kernel, 0, 0));
    // Round up according to array size
    gridSize_f2d = min((img_data.length + blockSize_f2d - 1ULL) / blockSize_f2d, 32ULL * numSMs);
    // Parameters for transpose_complex_matrix_kernel
    dim3 blockSize_tran(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));
    int gridSize_tran1_x = min((batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_tran1_y = min((img_data.length + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_tran1(gridSize_tran1_x, gridSize_tran1_y, 1);
    int gridSize_tran2_x = min((sf_data.num_lags + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_tran2_y = min((batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_tran2(gridSize_tran2_x, gridSize_tran2_y, 1);
    // Parameters for structure function
    int blockSize_corr = min(nextPowerOfTwo(img_data.length), 512ULL);
    int gridSize_corr_x = min((sf_data.num_lags + blockSize_corr - 1ULL) / blockSize_corr, (unsigned long long)maxGridSizeX);
    int gridSize_corr_y = min(batch_size, (unsigned long long)maxGridSizeY);
    dim3 gridSize_corr(gridSize_corr_x, gridSize_corr_y, 1);
    int smemSize = (blockSize_corr <= 32) ? 2ULL * blockSize_corr * sizeof(double) : 1ULL * blockSize_corr * sizeof(double);
    // Parameters for reduction (power spectrum and variance)
    int blockSize_red = min(nextPowerOfTwo(img_data.length), 512ULL);
    int gridSize_red = min(batch_size, (unsigned long long)maxGridSizeX);
    int smemSize2 = (blockSize_corr <= 32) ? 2ULL * blockSize_corr * sizeof(double2) : 1ULL * blockSize_corr * sizeof(double2);
    // Parameters for linear_combination_kernel
    int blockSize_lc; // The launch configurator returned block size
    int gridSize_lc;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_lc, linear_combination_kernel, 0, 0));
    // Round up according to array size
    gridSize_lc = min((batch_size + blockSize_lc - 1ULL) / blockSize_lc, 32ULL * numSMs);

    // *** Compute structure function
    for (unsigned long long batch = 0; batch < exec_params.num_batch_loops; batch++)
    {
        // Get start and end q point indices and current batch size
        unsigned long long q_start = batch * batch_size;
        unsigned long long q_end = (batch + 1ULL) * batch_size < sf_data.nx_half * sf_data.ny ? (batch + 1ULL) * batch_size : sf_data.nx_half * sf_data.ny;
        unsigned long long curr_batch_size = q_end - q_start;

        if (curr_batch_size != batch_size)
        {
            // If batch size changes, modify kernel execution parameters for transpose
            // This might be true only during last iteration
            gridSize_tran1.x = min((curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeX);
            gridSize_tran2.y = min((curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeY);
            gridSize_corr.y = min(curr_batch_size, (unsigned long long)maxGridSizeY);
        }

        // *** Copy values from host to device
        // Elements are complex Scalar
        // To speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D((Scalar *)d_workspace2,
                               2ULL * pitch_data.p_q * sizeof(Scalar),
                               h_in + 2ULL * q_start,
                               2ULL * sf_data.nx_half * sf_data.ny * sizeof(Scalar),
                               2ULL * curr_batch_size * sizeof(Scalar),
                               img_data.length,
                               cudaMemcpyHostToDevice));

#ifdef SINGLE_PRECISION
        // *** Convert data from float to double
        // Convert
        float2double_kernel<<<gridSize_f2d, blockSize_f2d>>>((float *)d_workspace2,
                                                             2ULL * pitch_data.p_q,
                                                             d_workspace1,
                                                             2ULL * pitch_data.p_q,
                                                             2ULL * curr_batch_size,
                                                             img_data.length);
        // Swap pointers
        swap(d_workspace1, d_workspace2);
#endif

        // *** Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran1, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_data.p_q,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_data.p_t,
                                                                            curr_batch_size,
                                                                            img_data.length,
                                                                            (curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM,
                                                                            (img_data.length + TILE_DIM - 1ULL) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Reset workspace2 to 0
        gpuErrchk(cudaMemset(d_workspace2, 0.0, workspace_size));

        // *** Compute structure function using differences (d_workspace1 --> d_workspace2)
        structure_function_diff_kernel<<<gridSize_corr, blockSize_corr, smemSize>>>((double2 *)d_workspace1,
                                                                                    (double2 *)d_workspace2,
                                                                                    d_lags,
                                                                                    img_data.length,
                                                                                    sf_data.num_lags,
                                                                                    curr_batch_size,
                                                                                    pitch_data.p_t);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Compute power spectrum (d_workspace1 --> d_power_spect)
        average_power_spectrum_kernel<<<gridSize_red, blockSize_red, smemSize>>>((double2 *)d_workspace1,
                                                                                 d_power_spec,
                                                                                 img_data.length,
                                                                                 pitch_data.p_t,
                                                                                 curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Compute variance (d_workspace1 --> d_var)
        // compute average over time
        average_complex_kernel<<<gridSize_red, blockSize_red, smemSize2>>>((double2 *)d_workspace1,
                                                                           d_var,
                                                                           img_data.length,
                                                                           pitch_data.p_t,
                                                                           curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // compute square modulus
        square_modulus_kernel<<<gridSize_lc, blockSize_lc>>>(d_var,
                                                             1,
                                                             1,
                                                             curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // linear combination (d_var = d_power_spec - d_var)
        linear_combination_kernel<<<gridSize_lc, blockSize_lc>>>(d_var,
                                                                 d_power_spec,
                                                                 make_double2(1.0, 0.0),
                                                                 d_var,
                                                                 make_double2(-1.0, 0.0),
                                                                 curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran2, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_data.p_t,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_data.p_q,
                                                                            sf_data.num_lags,
                                                                            curr_batch_size,
                                                                            (sf_data.num_lags + TILE_DIM - 1ULL) / TILE_DIM,
                                                                            (curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

#ifdef SINGLE_PRECISION
        // *** Convert data from double to float
        // Convert
        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>(d_workspace1,
                                                             2ULL * pitch_data.p_q,
                                                             (float *)d_workspace2,
                                                             2ULL * pitch_data.p_q,
                                                             2ULL * curr_batch_size,
                                                             img_data.length);
        // Swap pointers
        swap(d_workspace1, d_workspace2);
#endif

        // *** Copy values from device to host
        // Elements are treated as complex Scalar
        // To speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D(h_in + 2ULL * q_start,
                               2ULL * sf_data.nx_half * sf_data.ny * sizeof(Scalar),
                               (Scalar *)d_workspace1,
                               2ULL * pitch_data.p_q * sizeof(Scalar),
                               2ULL * curr_batch_size * sizeof(Scalar),
                               sf_data.num_lags,
                               cudaMemcpyDeviceToHost));

#ifndef SINGLE_PRECISION
        // copy power spectrum
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + sf_data.num_lags * sf_data.nx_half * sf_data.ny + q_start,
                             d_power_spec,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
        // copy variance
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (sf_data.num_lags + 1ULL) * sf_data.nx_half * sf_data.ny + q_start,
                             d_var,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
#else
        // *** Convert power spectrum and variance from double to float
        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>((double *)d_power_spec,
                                                             0,
                                                             (float *)d_workspace1,
                                                             0,
                                                             2ULL * curr_batch_size,
                                                             1);

        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>((double *)d_var,
                                                             0,
                                                             (float *)d_workspace2,
                                                             0,
                                                             2ULL * curr_batch_size,
                                                             1);

        // copy power spectrum
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + sf_data.num_lags * sf_data.nx_half * sf_data.ny + q_start,
                             (Scalar2 *)d_workspace1,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
        // copy variance
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (sf_data.num_lags + 1ULL) * sf_data.nx_half * sf_data.ny + q_start,
                             (Scalar2 *)d_workspace2,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
#endif
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
    gpuErrchk(cudaFree(d_lags));
    gpuErrchk(cudaFree(d_power_spec));
    gpuErrchk(cudaFree(d_var));
}

/*!
    Compute structure function using the Wiener-Khinchin theorem on the GPU
*/
void structure_function_fft(Scalar *h_in,
                            vector<unsigned int> lags,
                            ImageData &img_data,
                            StructureFunctionData &sf_data,
                            ExecutionParameters &exec_params,
                            PitchData &pitch_data)
{
    // Compute the number of q points in each batch
    unsigned long long batch_size = (sf_data.nx_half * sf_data.ny - 1ULL) / exec_params.num_batch_loops + 1ULL;

    // *** Allocate device arrays
    // Allocate workspaces
    // To speed up memory transfer and meet the requirements for coalesced memory access,
    // we space each subarray in workspace1 by pitch_nt (over t) and
    // we space each subarray in workspace2 by pitch_q (over q) and pitch_t (over t)
    double *d_workspace1, *d_workspace2;
    gpuErrchk(cudaMalloc(&d_workspace1, batch_size * pitch_data.p_nt * 2ULL * sizeof(double)));
    unsigned long long workspace2_size = max(batch_size * pitch_data.p_t, pitch_data.p_q * img_data.length);
    gpuErrchk(cudaMalloc(&d_workspace2, workspace2_size * 2ULL * sizeof(double)));
    // Allocate helper arrays
    unsigned int *d_lags;
    double2 *d_power_spec, *d_var;
    gpuErrchk(cudaMalloc(&d_lags, sf_data.num_lags * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_power_spec, batch_size * sizeof(double2)));
    gpuErrchk(cudaMalloc(&d_var, batch_size * sizeof(double2)));

    // Copy lags to device
    gpuErrchk(cudaMemcpy(d_lags, lags.data(), sf_data.num_lags * sizeof(unsigned int), cudaMemcpyHostToDevice));

    // *** Create the fft plan
    cufftHandle fft_plan = create_fft_plan(exec_params.nt, batch_size, pitch_data.p_nt);

    // *** Estimate efficient execution configuration
    // Get device id
    int device_id = get_device();
    // Get number of streaming multiprocessors
    int numSMs;
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id));
    // Parameters for float2double_kernel
    int minGridSize;   // The minimum grid size needed to achieve the
                       // maximum occupancy for a full device launch
    int blockSize_f2d; // The launch configurator returned block size
    int gridSize_f2d;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_f2d, float2double_kernel, 0, 0));
    // Round up according to array size
    gridSize_f2d = min((img_data.length + blockSize_f2d - 1) / blockSize_f2d, 32ULL * numSMs);
    // Parameters for transpose_complex_matrix_kernel
    dim3 blockSize_tran(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));
    int gridSize_tran1_x = min((batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_tran1_y = min((img_data.length + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_tran1(gridSize_tran1_x, gridSize_tran1_y, 1);
    int gridSize_tran2_x = min((sf_data.num_lags + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_tran2_y = min((batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_tran2(gridSize_tran2_x, gridSize_tran2_y, 1);
    // Parameters for square_modulus_kernel
    int blockSize_sqmod; // The launch configurator returned block size
    int gridSize_sqmod;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_sqmod, square_modulus_kernel, 0, 0));
    // Round up according to array size
    gridSize_sqmod = min((batch_size + blockSize_sqmod - 1ULL) / blockSize_sqmod, 32ULL * numSMs);
    // Parameters for scale_array_kernel
    int blockSize_scale; // The launch configurator returned block size
    int gridSize_scale;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_scale, scale_array_kernel, 0, 0));
    // Round up according to array size
    gridSize_scale = min((batch_size + blockSize_scale - 1ULL) / blockSize_scale, 32ULL * numSMs);
    // Parameters for linear_combination_final_kernel
    int blockSize_final; // The launch configurator returned block size
    int gridSize_final;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_final, linear_combination_final_kernel, 0, 0));
    // Round up according to array size
    gridSize_final = min(batch_size, 32ULL * numSMs);
    // Parameters for copy selected lags kernel
    int blockSize_copy; // The launch configurator returned block size
    int gridSize_copy;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_copy, copy_selected_lags_kernel, 0, 0));
    // Round up according to array size
    gridSize_copy = min(batch_size, 32ULL * numSMs);
    // Parameters for linear combination of power spectrum and variance
    int blockSize_lc; // The launch configurator returned block size
    int gridSize_lc;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_lc, linear_combination_kernel, 0, 0));
    // Round up according to array size
    gridSize_lc = min((batch_size + blockSize_lc - 1ULL) / blockSize_lc, 32ULL * numSMs);

    // *** Compute structure function
    for (unsigned long long batch = 0; batch < exec_params.num_batch_loops; batch++)
    {
        // Get start and end q point indices and current batch size
        unsigned long long q_start = batch * batch_size;
        unsigned long long q_end = (batch + 1ULL) * batch_size < sf_data.nx_half * sf_data.ny ? (batch + 1ULL) * batch_size : sf_data.nx_half * sf_data.ny;
        unsigned long long curr_batch_size = q_end - q_start;

        if (curr_batch_size != batch_size)
        {
            // If batch size changes, modify kernel execution parameters for transpose and linear combination
            // and create a new fft plan
            // This might be true only during last iteration
            cufftSafeCall(cufftDestroy(fft_plan));
            fft_plan = create_fft_plan(exec_params.nt,
                                       curr_batch_size,
                                       pitch_data.p_nt);
            gridSize_tran1.x = min((curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeX);
            gridSize_tran2.y = min((curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM, (unsigned long long)maxGridSizeY);
            gridSize_final = min(curr_batch_size, 32ULL * numSMs);
            gridSize_sqmod = min((curr_batch_size + blockSize_sqmod - 1ULL) / blockSize_sqmod, 32ULL * numSMs);
            gridSize_scale = min((curr_batch_size + blockSize_scale - 1ULL) / blockSize_scale, 32ULL * numSMs);
        }

        // *** Reset workspace1 and workspace2 to 0
        gpuErrchk(cudaMemset2D(d_workspace1,
                               2ULL * pitch_data.p_nt * sizeof(double),
                               0.0,
                               2ULL * pitch_data.p_nt * sizeof(double),
                               curr_batch_size));
        gpuErrchk(cudaMemset(d_workspace2, 0.0, 2ULL * workspace2_size * sizeof(double)));

        // *** Copy values from host to device
        // Elements are complex Scalar
        // To speed up transfer, use pitch_q
        unsigned long long offset = 2ULL * q_start;                                         // host source array offset
        unsigned long long spitch = 2ULL * (sf_data.nx_half * sf_data.ny) * sizeof(Scalar); // host source array pitch
        unsigned long long dpitch = 2ULL * pitch_data.p_q * sizeof(Scalar);                 // device destination array pitch
#ifndef SINGLE_PRECISION
        gpuErrchk(cudaMemcpy2D((Scalar *)d_workspace2,
                               dpitch,
                               h_in + offset,
                               spitch,
                               2ULL * curr_batch_size * sizeof(Scalar),
                               img_data.length,
                               cudaMemcpyHostToDevice));
#else
        gpuErrchk(cudaMemcpy2D((Scalar *)d_workspace1,
                               dpitch,
                               h_in + offset,
                               spitch,
                               2ULL * curr_batch_size * sizeof(Scalar),
                               img_data.length,
                               cudaMemcpyHostToDevice));

        // ***Convert data from float to double
        float2double_kernel<<<gridSize_f2d, blockSize_f2d>>>((float *)d_workspace1,
                                                             2ULL * pitch_data.p_q,
                                                             d_workspace2,
                                                             2ULL * pitch_data.p_q,
                                                             2ULL * curr_batch_size,
                                                             img_data.length);

        // Reset again d_workspace1 to 0
        gpuErrchk(cudaMemset2D(d_workspace1, 2ULL * pitch_data.p_nt * sizeof(double), 0.0, 2ULL * exec_params.nt * sizeof(double), curr_batch_size));
#endif

        // ***Transpose complex matrix ({d_workspace2; pitch_q} --> {d_workspace1; pitch_nt})
        transpose_complex_matrix_kernel<<<gridSize_tran1, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_data.p_q,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_data.p_nt,
                                                                            curr_batch_size,
                                                                            img_data.length,
                                                                            (curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM,
                                                                            (img_data.length + TILE_DIM - 1ULL) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy values ({d_workspace1; pitch_nt} --> {d_workspace2; pitch_t})
        dpitch = 2ULL * pitch_data.p_t * sizeof(double);
        spitch = 2ULL * pitch_data.p_nt * sizeof(double);
        gpuErrchk(cudaMemcpy2D(d_workspace2,
                               dpitch,
                               d_workspace1,
                               spitch,
                               img_data.length * 2ULL * sizeof(double),
                               curr_batch_size,
                               cudaMemcpyDeviceToDevice));

        // +++ FFT PART +++
        // ***Do fft (d_workspace1 --> d_workspace1)
        cufftSafeCall(cufftExecZ2Z(fft_plan, (CUFFTCOMPLEX *)d_workspace1, (CUFFTCOMPLEX *)d_workspace1, CUFFT_FORWARD));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Compute square modulus (d_workspace1 --> d_workspace1)
        square_modulus_kernel<<<gridSize_sqmod, blockSize_sqmod>>>((double2 *)d_workspace1,
                                                                   exec_params.nt,
                                                                   pitch_data.p_nt,
                                                                   curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy square modulus of sum part
        gpuErrchk(cudaMemcpy2D(d_var,
                               sizeof(double2),
                               d_workspace1,
                               pitch_data.p_nt * sizeof(double2),
                               sizeof(double2),
                               curr_batch_size,
                               cudaMemcpyDeviceToDevice));

        // ***Do fft (d_workspace1 --> d_workspace1)
        cufftSafeCall(cufftExecZ2Z(fft_plan, (CUFFTCOMPLEX *)d_workspace1, (CUFFTCOMPLEX *)d_workspace1, CUFFT_FORWARD));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Scale fft part (d_workspace1 --> d_workspace1)
        scale_array_kernel<<<gridSize_scale, blockSize_scale>>>((CUFFTCOMPLEX *)d_workspace1,
                                                                pitch_data.p_nt,
                                                                exec_params.nt,
                                                                1.0 / (double)exec_params.nt,
                                                                curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy power spectrum part
        gpuErrchk(cudaMemcpy2D(d_power_spec,
                               sizeof(double2),
                               d_workspace1,
                               pitch_data.p_nt * sizeof(double2),
                               sizeof(double2),
                               curr_batch_size,
                               cudaMemcpyDeviceToDevice));

        // scale power spectrum (d_power_spec = d_power_spec * 1/length + 0)
        linear_combination_kernel<<<gridSize_lc, blockSize_lc>>>(d_power_spec,
                                                                 d_power_spec,
                                                                 make_double2(1.0 / (double)img_data.length, 0.0),
                                                                 d_power_spec,
                                                                 make_double2(0.0, 0.0),
                                                                 curr_batch_size);

        // ***Compute variance (d_var = (1.0, 0.0) * d_power_spec + (-1.0/(length * length), 0.0) * d_var)
        linear_combination_kernel<<<gridSize_lc, blockSize_lc>>>(d_var,
                                                                 d_power_spec,
                                                                 make_double2(1.0, 0.0),
                                                                 d_var,
                                                                 make_double2(-1.0 / (double)(img_data.length * img_data.length), 0.0),
                                                                 curr_batch_size);

        // +++ CUMULATIVE SUM PART +++
        // ***Compute square modulus (d_workspace2 --> d_workspace2)
        square_modulus_kernel<<<gridSize_sqmod, blockSize_sqmod>>>((double2 *)d_workspace2,
                                                                   img_data.length,
                                                                   pitch_data.p_t,
                                                                   curr_batch_size);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy the value into the imaginary part of the opposite (with respect to time) element (d_workspace2 --> d_workspace2)
        real2imagopposite_kernel<<<gridSize_sqmod, blockSize_sqmod>>>((CUFFTCOMPLEX *)d_workspace2,
                                                                      img_data.length,
                                                                      pitch_data.p_t,
                                                                      curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Compute (exclusive) cumulative sum (prefix scan)
        scan_wrap(d_workspace2,
                  d_workspace2,
                  2ULL * img_data.length,
                  2ULL * pitch_data.p_t,
                  curr_batch_size);

        //  ***Linearly combine the two parts (workspace1 + workspace2 --> workspace2)
        linear_combination_final_kernel<<<gridSize_final, blockSize_final>>>((double2 *)d_workspace2,
                                                                             pitch_data.p_t,
                                                                             (double2 *)d_workspace1,
                                                                             pitch_data.p_nt,
                                                                             (double2 *)d_workspace2,
                                                                             pitch_data.p_t,
                                                                             img_data.length,
                                                                             curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Keep only selected lags ({d_workspace2; pitch_t} --> {d_workspace1; pitch_t})
        copy_selected_lags_kernel<<<gridSize_copy, blockSize_copy>>>((double2 *)d_workspace2,
                                                                     (double2 *)d_workspace1,
                                                                     d_lags,
                                                                     sf_data.num_lags,
                                                                     pitch_data.p_t,
                                                                     pitch_data.p_t,
                                                                     curr_batch_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Transpose complex matrix ({d_workspace1; pitch_t} --> {d_workspace2; pitch_q})
        transpose_complex_matrix_kernel<<<gridSize_tran2, blockSize_tran>>>((double2 *)d_workspace1,
                                                                            pitch_data.p_t,
                                                                            (double2 *)d_workspace2,
                                                                            pitch_data.p_q,
                                                                            sf_data.num_lags,
                                                                            curr_batch_size,
                                                                            (sf_data.num_lags + TILE_DIM - 1ULL) / TILE_DIM,
                                                                            (curr_batch_size + TILE_DIM - 1ULL) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

#ifndef SINGLE_PRECISION
        // ***Copy from device to host (d_workspace2 --> host)
        // elements are treated as complex Scalars
        // to speed up transfer, use pitch_q
        offset = 2ULL * q_start;                                         // host destination array offset
        spitch = 2ULL * pitch_data.p_q * sizeof(Scalar);                 // host source array pitch
        dpitch = 2ULL * (sf_data.nx_half * sf_data.ny) * sizeof(Scalar); // device destination array pitch
        gpuErrchk(cudaMemcpy2D(h_in + offset,
                               dpitch,
                               (Scalar *)d_workspace2,
                               spitch,
                               2ULL * curr_batch_size * sizeof(Scalar),
                               sf_data.num_lags,
                               cudaMemcpyDeviceToHost));
#else
        // ***Convert data from double to float
        // Convert
        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>(d_workspace2,
                                                             2ULL * pitch_data.p_q,
                                                             (float *)d_workspace1,
                                                             2ULL * pitch_data.p_q,
                                                             2ULL * curr_batch_size,
                                                             img_data.length);

        // ***Copy from device to host (d_workspace1 --> host)
        // elements are treated as complex Scalars
        // to speed up transfer, use pitch_q
        offset = 2ULL * q_start;                                         // host destination array offset
        spitch = 2ULL * pitch_data.p_q * sizeof(Scalar);                 // host source array pitch
        dpitch = 2ULL * (sf_data.nx_half * sf_data.ny) * sizeof(Scalar); // device destination array pitch
        gpuErrchk(cudaMemcpy2D(h_in + offset,
                               dpitch,
                               (Scalar *)d_workspace1,
                               spitch,
                               2ULL * curr_batch_size * sizeof(Scalar),
                               sf_data.num_lags,
                               cudaMemcpyDeviceToHost));
#endif

#ifndef SINGLE_PRECISION
        // copy power spectrum
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + sf_data.num_lags * sf_data.nx_half * sf_data.ny + q_start,
                             d_power_spec,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));

        // copy variance
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (sf_data.num_lags + 1ULL) * sf_data.nx_half * sf_data.ny + q_start,
                             d_var,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
#else
        // ***Convert power spectrum and variance from double to float
        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>((double *)d_power_spec,
                                                             0,
                                                             (float *)d_workspace1,
                                                             0,
                                                             2ULL * curr_batch_size,
                                                             1);

        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>((double *)d_var,
                                                             0,
                                                             (float *)d_workspace2,
                                                             0,
                                                             2ULL * curr_batch_size,
                                                             1);

        // copy power spectrum
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + sf_data.num_lags * sf_data.nx_half * sf_data.ny + q_start,
                             (Scalar2 *)d_workspace1,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));

        // copy variance
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (sf_data.num_lags + 1ULL) * sf_data.nx_half * sf_data.ny + q_start,
                             (Scalar2 *)d_workspace2,
                             curr_batch_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
#endif
    }

    // ***Free memory
    cufftSafeCall(cufftDestroy(fft_plan));
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
    gpuErrchk(cudaFree(d_lags));
    gpuErrchk(cudaFree(d_power_spec));
    gpuErrchk(cudaFree(d_var));
}

/*!
    Convert to fftshifted structure function on the GPU
 */
void make_shift(Scalar *h_in,
                ImageData &img_data,
                StructureFunctionData &sf_data,
                ExecutionParameters &exec_params,
                PitchData &pitch_data)
{
    // Compute number of lags in a batch
    unsigned long long batch_size = (sf_data.length - 1ULL) / exec_params.num_fftshift_loops + 1ULL;

    // *** Allocate device arrays
    // Allocate workspaces
    Scalar *d_workspace1, *d_workspace2;
    gpuErrchk(cudaMalloc(&d_workspace1, pitch_data.p_fftshift * sf_data.ny * batch_size * 2ULL * sizeof(Scalar)));
    gpuErrchk(cudaMalloc(&d_workspace2, pitch_data.p_fftshift * sf_data.ny * batch_size * sizeof(Scalar)));

    // *** Estimate efficient execution configuration
    // Get device id
    int device_id = get_device();
    // Parameters for shift_powerspec_kernel
    dim3 blockSize_full(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));
    int gridSize_shift_x = min((sf_data.nx_half + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_shift_y = min((sf_data.ny * batch_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_shift(gridSize_shift_x, gridSize_shift_y, 1);

    // *** Perform fftshift
    for (unsigned long long batch = 0; batch < exec_params.num_fftshift_loops; batch++)
    {
        // Get input offset
        unsigned long long ioffset = batch * batch_size * 2ULL * sf_data.nx_half * sf_data.ny;
        // Get current batch size
        unsigned long long curr_batch_size = (batch + 1) * batch_size > sf_data.length ? sf_data.length - batch * batch_size : batch_size;

        if (curr_batch_size != batch_size)
        {
            // if batch size changes, modify kernel execution parameters
            gridSize_shift_y = min((sf_data.ny * curr_batch_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
            gridSize_shift.y = gridSize_shift_y;
        }

        // *** Copy values from host to device
        gpuErrchk(cudaMemcpy2D(d_workspace1,
                               pitch_data.p_fftshift * 2 * sizeof(Scalar),
                               h_in + ioffset,
                               2 * sf_data.nx_half * sizeof(Scalar),
                               2 * sf_data.nx_half * sizeof(Scalar),
                               curr_batch_size * sf_data.ny,
                               cudaMemcpyHostToDevice));

        // *** Shift power spectrum (workspace2 --> workspace1)
        shift_powspec_kernel<<<gridSize_shift, blockSize_full>>>((Scalar2 *)d_workspace1,
                                                                 pitch_data.p_fftshift,
                                                                 d_workspace2,
                                                                 pitch_data.p_fftshift,
                                                                 sf_data.nx_half,
                                                                 sf_data.ny,
                                                                 curr_batch_size,
                                                                 (sf_data.nx_half + TILE_DIM - 1) / TILE_DIM,
                                                                 (sf_data.ny * curr_batch_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy values from device to host (make contiguous on memory)
        // Get output offset
        unsigned long long ooffset = batch * batch_size * sf_data.nx_half * sf_data.ny;
        gpuErrchk(cudaMemcpy2D(h_in + ooffset,
                               sf_data.nx_half * sizeof(Scalar),
                               d_workspace2,
                               pitch_data.p_fftshift * sizeof(Scalar),
                               sf_data.nx_half * sizeof(Scalar),
                               curr_batch_size * sf_data.ny,
                               cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
}
