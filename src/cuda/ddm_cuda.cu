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
                  bool is_window,
                  unsigned long long width,
                  unsigned long long height,
                  unsigned long long length,
                  unsigned long long nx,
                  unsigned long long ny,
                  unsigned long long num_fft2,
                  unsigned long long buff_pitch,
                  unsigned long long pitch_nx)
{
    // Compute the width of the real to complex fft2
    unsigned long long _nx = nx / 2 + 1;
    // Compute the number of batched fft2
    unsigned long long batch = (length - 1) / num_fft2 + 1;
    // Compute the fft2 normalization factor
    double norm_fact = 1.0 / sqrt((double)(nx * ny));

    // *** Allocate device arrays
    // Allocate workspace
    double *d_workspace;
    gpuErrchk(cudaMalloc(&d_workspace, 2 * pitch_nx * ny * batch * sizeof(double)));
    // Allocate buffer space (only if input type is not double already)
    T *d_buff;
    if (!std::is_same<T, double>::value)
    {
        gpuErrchk(cudaMalloc(&d_buff, buff_pitch * height * batch * sizeof(T)));
    }
    // Allocate window space (only if window is given by the user)
    Scalar *d_window;
    if (is_window)
    {
        gpuErrchk(cudaMalloc(&d_window, 2 * pitch_nx * ny * sizeof(double)));
    }

    // *** Create the fft2 plan
    cufftHandle fft2_plan = create_fft2_plan(nx, ny, batch, pitch_nx);

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
    dim3 gridSize_copy(min(1ULL * maxGridSizeX, batch), min(1ULL * maxGridSizeY, height), 1);
    // Parameters for apply_window_kernel
    int blockSize_win;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_win, apply_window_kernel, 0, 0));
    dim3 gridSize_win(min(1ULL * maxGridSizeX, batch), min(1ULL * maxGridSizeY, height), 1);
    // Parameters for scale_array_kernel
    int blockSize_scale; // The launch configurator returned block size
    int gridSize_scale;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_scale, scale_array_kernel, 0, 0));
    // Round up according to array size
    gridSize_scale = min((ny * batch + blockSize_scale - 1) / blockSize_scale, 32ULL * numSMs);

#ifdef SINGLE_PRECISION
    // in place conversion kernel
    int smem_size = _nx * sizeof(float2);
#endif

    // *** Compute fft2
    for (unsigned long long ii = 0; ii < num_fft2; ii++)
    {
        // *** Reset workspace array to 0 (ensures zero padding)
        gpuErrchk(cudaMemset(d_workspace, 0.0, 2 * pitch_nx * ny * batch * sizeof(double)));

        // *** Transfer images to GPU
        // If input is double, transfer memory directly with pitch
        if (std::is_same<T, double>::value)
        {
            // Compute the number of images to copy
            unsigned long long num_imgs_copy = batch;
            if ((ii + 1) * batch > length)
            {
                num_imgs_copy = length - ii * batch;
            }

            // Copy with cudaMemcpy3D
            cudaMemcpy3DParms params = {0};
            params.srcArray = NULL;
            params.srcPos = make_cudaPos(0, 0, ii * batch);
            params.srcPtr = make_cudaPitchedPtr((double *)h_in, width * sizeof(double), width, height);
            params.dstArray = NULL;
            params.dstPos = make_cudaPos(0, 0, 0);
            params.dstPtr = make_cudaPitchedPtr(d_workspace, 2 * pitch_nx * sizeof(double), 2 * pitch_nx, ny);
            params.extent = make_cudaExtent(width * sizeof(double), height, num_imgs_copy);
            params.kind = cudaMemcpyHostToDevice;

            gpuErrchk(cudaMemcpy3D(&params));
        }
        else // If input is not double, transfer memory to buffer and convert to double to workspace
        {
            // Reset buffer array to 0
            gpuErrchk(cudaMemset(d_buff, (T)0, buff_pitch * height * batch * sizeof(T)));

            // Compute offset index (starting index of the current batch)
            unsigned long long offset = ii * width * height * batch;
            // Compute number of rows to copy
            unsigned long long num_rows_copy = height * batch;
            if ((ii + 1) * batch > length)
            {
                num_rows_copy = height * (length - ii * batch);
            }
            // Copy values to buffer
            gpuErrchk(cudaMemcpy2D(d_buff, buff_pitch * sizeof(T), h_in + offset, width * sizeof(T), width * sizeof(T), num_rows_copy, cudaMemcpyHostToDevice));

            // Convert values of buffer into workspace
            copy_convert_kernel<<<gridSize_copy, blockSize_copy>>>(d_buff,
                                                                   d_workspace,
                                                                   width,
                                                                   height,
                                                                   batch,
                                                                   buff_pitch,
                                                                   buff_pitch * height,
                                                                   2 * pitch_nx,
                                                                   2 * pitch_nx * ny);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        // *** Apply the window function
        if (is_window)
        {
            // copy window function to device
            gpuErrchk(cudaMemcpy2D(d_window, 2 * pitch_nx * sizeof(Scalar), h_window, width * sizeof(Scalar), width * sizeof(Scalar), height, cudaMemcpyHostToDevice));
            // apply window
            apply_window_kernel<<<gridSize_win, blockSize_win>>>(d_workspace,
                                                                 d_window,
                                                                 width,
                                                                 height,
                                                                 batch,
                                                                 2 * pitch_nx,
                                                                 2 * pitch_nx * ny,
                                                                 2 * pitch_nx);
            gpuErrchk(cudaPeekAtLastError());
            gpuErrchk(cudaDeviceSynchronize());
        }

        // ***Execute fft2 plan
        cufftSafeCall(cufftExecD2Z(fft2_plan, d_workspace, (CUFFTCOMPLEX *)d_workspace));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Normalize fft2
        // Starting index
        unsigned long long start = ii * ny * batch;
        // Final index (if exceeds array size, truncate)
        unsigned long long end = (ii + 1) * batch > length ? length * ny : (ii + 1) * ny * batch;
        // scale array
        scale_array_kernel<<<gridSize_scale, blockSize_scale>>>((double2 *)d_workspace,
                                                                pitch_nx,
                                                                _nx,
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
                                                                                    pitch_nx,
                                                                                    _nx,
                                                                                    end - start);
#endif

        // ***Copy values back to host
        gpuErrchk(cudaMemcpy2D((Scalar2 *)h_out + _nx * start, _nx * sizeof(Scalar2), (Scalar2 *)d_workspace, pitch_nx * sizeof(double2), _nx * sizeof(Scalar2), end - start, cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace));
    if (!std::is_same<T, double>::value)
    {
        gpuErrchk(cudaFree(d_buff));
    }
    if (is_window)
    {
        gpuErrchk(cudaFree(d_window));
    }
    cufftSafeCall(cufftDestroy(fft2_plan));
}

template void compute_fft2<uint8_t>(const uint8_t *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<int16_t>(const int16_t *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<uint16_t>(const uint16_t *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<int32_t>(const int32_t *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<uint32_t>(const uint32_t *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<int64_t>(const int64_t *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<uint64_t>(const uint64_t *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<float>(const float *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<double>(const double *h_in, Scalar *h_out, const Scalar *h_window, bool is_window, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);

/*!
    Compute image structure function using differences on the GPU
 */
void structure_function_diff(Scalar *h_in,
                             vector<unsigned int> lags,
                             unsigned long long length,
                             unsigned long long nx,
                             unsigned long long ny,
                             unsigned long long num_chunks,
                             unsigned long long pitch_q,
                             unsigned long long pitch_t)
{
    // Compute the width of the real to complex fft2
    unsigned long long _nx = nx / 2 + 1;
    // Compute the number of q points in each chunk
    unsigned long long chunk_size = (_nx * ny - 1) / num_chunks + 1;

    // *** Allocate device arrays
    // Allocateworkspaces
    double *d_workspace1, *d_workspace2;
    unsigned long long workspace_size = max(pitch_q * length, chunk_size * pitch_t) * 2 * sizeof(double);
    gpuErrchk(cudaMalloc(&d_workspace1, workspace_size));
    gpuErrchk(cudaMalloc(&d_workspace2, workspace_size));
    // Allocate helper arrays
    unsigned int *d_lags;
    double2 *d_power_spec, *d_var;
    gpuErrchk(cudaMalloc(&d_lags, lags.size() * sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_power_spec, chunk_size * sizeof(double2)));
    gpuErrchk(cudaMalloc(&d_var, chunk_size * sizeof(double2)));

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
    gridSize_f2d = min((length + blockSize_f2d - 1) / blockSize_f2d, 32ULL * numSMs);
    // Parameters for transpose_complex_matrix_kernel
    dim3 blockSize_tran(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));
    int gridSize_tran1_x = min((chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_tran1_y = min((length + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_tran1(gridSize_tran1_x, gridSize_tran1_y, 1);
    int gridSize_tran2_x = min(((unsigned long long)(lags.size()) + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_tran2_y = min((chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_tran2(gridSize_tran2_x, gridSize_tran2_y, 1);
    // Parameters for structure function
    int blockSize_corr = min(nextPowerOfTwo(length), 512ULL);
    int gridSize_corr_x = min(((unsigned long long)(lags.size()) + blockSize_corr - 1) / blockSize_corr, (unsigned long long)maxGridSizeX);
    int gridSize_corr_y = min(chunk_size, (unsigned long long)maxGridSizeY);
    dim3 gridSize_corr(gridSize_corr_x, gridSize_corr_y, 1);
    int smemSize = (blockSize_corr <= 32) ? 2ULL * blockSize_corr * sizeof(double) : 1ULL * blockSize_corr * sizeof(double);
    // Parameters for reduction (power spectrum and variance)
    int blockSize_red = min(nextPowerOfTwo(length), 512ULL);
    int gridSize_red = min(chunk_size, (unsigned long long)maxGridSizeX);
    int smemSize2 = (blockSize_corr <= 32) ? 2ULL * blockSize_corr * sizeof(double2) : 1ULL * blockSize_corr * sizeof(double2);
    // Parameters for linear_combination_kernel
    int blockSize_lc; // The launch configurator returned block size
    int gridSize_lc;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_lc, linear_combination_kernel, 0, 0));
    // Round up according to array size
    gridSize_lc = min((chunk_size + blockSize_lc - 1) / blockSize_lc, 32ULL * numSMs);

    // *** Compute structure function
    for (unsigned long long chunk = 0; chunk < num_chunks; chunk++)
    {
        // Get start and end q point indices and current chunk size
        unsigned long long q_start = chunk * chunk_size;
        unsigned long long q_end = (chunk + 1) * chunk_size < _nx * ny ? (chunk + 1) * chunk_size : _nx * ny;
        unsigned long long curr_chunk_size = q_end - q_start;

        if (curr_chunk_size != chunk_size)
        {
            // if chunk size changes, modify kernel execution parameters for transpose
            gridSize_tran1.x = min((curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeX);
            gridSize_tran2.y = min((curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
            gridSize_corr.y = min(curr_chunk_size, (unsigned long long)maxGridSizeY);
        }

        // *** Copy values from host to device
        // Elements are complex Scalar
        // To speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D((Scalar *)d_workspace2,
                               2 * pitch_q * sizeof(Scalar),
                               h_in + 2 * q_start,
                               2 * _nx * ny * sizeof(Scalar),
                               2 * curr_chunk_size * sizeof(Scalar),
                               length,
                               cudaMemcpyHostToDevice));

#ifdef SINGLE_PRECISION
        // *** Convert data from float to double
        // Convert
        float2double_kernel<<<gridSize_f2d, blockSize_f2d>>>((float *)d_workspace2,
                                                             2 * pitch_q,
                                                             d_workspace1,
                                                             2 * pitch_q,
                                                             2 * curr_chunk_size,
                                                             length);
        // Swap pointers
        swap(d_workspace1, d_workspace2);
#endif

        // *** Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran1, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_q,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_t,
                                                                            curr_chunk_size,
                                                                            length,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM,
                                                                            (length + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Reset workspace2 to 0
        gpuErrchk(cudaMemset(d_workspace2, 0.0, workspace_size));

        // *** Compute structure function using differences (d_workspace1 --> d_workspace2)
        structure_function_diff_kernel<<<gridSize_corr, blockSize_corr, smemSize>>>((double2 *)d_workspace1,
                                                                                    (double2 *)d_workspace2,
                                                                                    d_lags,
                                                                                    length,
                                                                                    lags.size(),
                                                                                    curr_chunk_size,
                                                                                    pitch_t);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Compute power spectrum (d_workspace1 --> d_power_spect)
        average_power_spectrum_kernel<<<gridSize_red, blockSize_red, smemSize>>>((double2 *)d_workspace1,
                                                                                 d_power_spec,
                                                                                 length,
                                                                                 pitch_t,
                                                                                 curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Compute variance (d_workspace1 --> d_var)
        // compute average over time
        average_complex_kernel<<<gridSize_red, blockSize_red, smemSize2>>>((double2 *)d_workspace1,
                                                                           d_var,
                                                                           length,
                                                                           pitch_t,
                                                                           curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // compute square modulus
        square_modulus_kernel<<<gridSize_lc, blockSize_lc>>>(d_var,
                                                             1,
                                                             1,
                                                             curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // linear combination (d_var = d_power_spec - d_var)
        linear_combination_kernel<<<gridSize_lc, blockSize_lc>>>(d_var,
                                                                 d_power_spec,
                                                                 make_double2(1.0, 0.0),
                                                                 d_var,
                                                                 make_double2(-1.0, 0.0),
                                                                 curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // *** Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran2, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_t,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_q,
                                                                            lags.size(),
                                                                            curr_chunk_size,
                                                                            (lags.size() + TILE_DIM - 1) / TILE_DIM,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

#ifdef SINGLE_PRECISION
        // *** Convert data from double to float
        // Convert
        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>(d_workspace1,
                                                             2 * pitch_q,
                                                             (float *)d_workspace2,
                                                             2 * pitch_q,
                                                             2 * curr_chunk_size,
                                                             length);
        // Swap pointers
        swap(d_workspace1, d_workspace2);
#endif

        // *** Copy values from device to host
        // Elements are treated as complex Scalar
        // To speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D(h_in + 2 * q_start,
                               2 * _nx * ny * sizeof(Scalar),
                               (Scalar *)d_workspace1,
                               2 * pitch_q * sizeof(Scalar),
                               2 * curr_chunk_size * sizeof(Scalar),
                               lags.size(),
                               cudaMemcpyDeviceToHost));

#ifndef SINGLE_PRECISION
        // copy power spectrum
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (unsigned long long)(lags.size()) * _nx * ny + q_start,
                             d_power_spec,
                             curr_chunk_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
        // copy variance
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (unsigned long long)(lags.size() + 1) * _nx * ny + q_start,
                             d_var,
                             curr_chunk_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
#else
        // *** Convert power spectrum and variance from double to float
        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>((double *)d_power_spec,
                                                             0,
                                                             (float *)d_workspace1,
                                                             0,
                                                             2 * curr_chunk_size,
                                                             1);

        double2float_kernel<<<gridSize_f2d, blockSize_f2d>>>((double *)d_var,
                                                             0,
                                                             (float *)d_workspace2,
                                                             0,
                                                             2 * curr_chunk_size,
                                                             1);

        // copy power spectrum
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (unsigned long long)(lags.size()) * _nx * ny + q_start,
                             (Scalar2 *)d_workspace1,
                             curr_chunk_size * sizeof(Scalar2),
                             cudaMemcpyDeviceToHost));
        // copy variance
        gpuErrchk(cudaMemcpy((Scalar2 *)h_in + (unsigned long long)(lags.size() + 1) * _nx * ny + q_start,
                             (Scalar2 *)d_workspace2,
                             curr_chunk_size * sizeof(Scalar2),
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

/*! \brief Convert to fftshifted image structure function on the GPU
    \param h_in             input array after structure function calculation
    \param Nlags            number of lags analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param num_shift        number of shift chunks
    \param pitch_fs         pitch of device array for shift operations
 */
void make_shift(Scalar *h_in,
                unsigned long long Nlags,
                unsigned long long nx,
                unsigned long long ny,
                unsigned long long num_shift,
                unsigned long long pitch_fs)
{
    // Compute the width of the real to complex fft2
    unsigned long long _nx = nx / 2 + 1;
    // Compute number of lags in a chunk
    unsigned long long chunk_size = (Nlags - 1) / num_shift + 1;

    // *** Allocate device arrays
    // Allocate workspaces
    Scalar *d_workspace1, *d_workspace2;
    gpuErrchk(cudaMalloc(&d_workspace1, pitch_fs * ny * chunk_size * 2 * sizeof(Scalar)));
    gpuErrchk(cudaMalloc(&d_workspace2, pitch_fs * ny * chunk_size * sizeof(Scalar)));

    // *** Estimate efficient execution configuration
    // Get device id
    int device_id = get_device();
    // Parameters for shift_powerspec_kernel
    dim3 blockSize_full(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));
    int gridSize_shift_x = min((_nx + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_shift_y = min((ny * chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_shift(gridSize_shift_x, gridSize_shift_y, 1);

    // *** Perform fftshift
    for (unsigned long long chunk = 0; chunk < num_shift; chunk++)
    {
        // Get input offset
        unsigned long long ioffset = chunk * chunk_size * 2 * _nx * ny;
        // Get current chunk size
        unsigned long long curr_chunk_size = (chunk + 1) * chunk_size > Nlags ? Nlags - chunk * chunk_size : chunk_size;

        if (curr_chunk_size != chunk_size)
        {
            // if chunk size changes, modify kernel execution parameters
            gridSize_shift_y = min((ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
            gridSize_shift.y = gridSize_shift_y;
        }

        // *** Copy values from host to device
        gpuErrchk(cudaMemcpy2D(d_workspace1,
                               pitch_fs * 2 * sizeof(Scalar),
                               h_in + ioffset,
                               2 * _nx * sizeof(Scalar),
                               2 * _nx * sizeof(Scalar),
                               curr_chunk_size * ny,
                               cudaMemcpyHostToDevice));

        // *** Shift power spectrum (workspace2 --> workspace1)
        shift_powspec_kernel<<<gridSize_shift, blockSize_full>>>((Scalar2 *)d_workspace1,
                                                                 pitch_fs,
                                                                 d_workspace2,
                                                                 pitch_fs,
                                                                 _nx,
                                                                 ny,
                                                                 curr_chunk_size,
                                                                 (_nx + TILE_DIM - 1) / TILE_DIM,
                                                                 (ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy values from device to host (make contiguous on memory)
        // Get output offset
        unsigned long long ooffset = chunk * chunk_size * _nx * ny;
        gpuErrchk(cudaMemcpy2D(h_in + ooffset,
                               _nx * sizeof(Scalar),
                               d_workspace2,
                               pitch_fs * sizeof(Scalar),
                               _nx * sizeof(Scalar),
                               curr_chunk_size * ny,
                               cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
}
