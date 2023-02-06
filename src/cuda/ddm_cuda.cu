// Maintainer: enrico-lattuada

/*! \file ddm_cuda.cu
    \brief Definition of core CUDA Differential Dynamic Microscopy functions
*/

// *** headers ***
#include "ddm_cuda.cuh"

#include "helper_debug.cuh"
#include "helper_cufft.cuh"
#include "helper_ddm_cuda.cuh"
#include "helper_prefix_sum.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdlib.h>

// #include <chrono>
// using namespace std::chrono;
// typedef std::chrono::high_resolution_clock Time;
// typedef std::chrono::duration<float> fsec;

// auto t0 = Time::now();
// auto t1 = Time::now();
// fsec fs = t0 - t0;
// fprintf(stdout, "%f\n", fs.count());

#ifndef SINGLE_PRECISION
typedef cufftDoubleComplex CUFFTCOMPLEX;
typedef double2 Scalar2;
cufftResult (*CufftExecR2C)(cufftHandle, cufftDoubleReal*, cufftDoubleComplex*) = &cufftExecD2Z;
cufftResult (*CufftExecC2C)(cufftHandle, cufftDoubleComplex*, cufftDoubleComplex*, int) = &cufftExecZ2Z;
double2 (*make_Scalar2)(double, double) = &make_double2;
#else
typedef cufftComplex CUFFTCOMPLEX;
typedef float2 Scalar2;
cufftResult (*CufftExecR2C)(cufftHandle, cufftReal*, cufftComplex*) = &cufftExecR2C;
cufftResult (*CufftExecC2C)(cufftHandle, cufftComplex*, cufftComplex*, int) = &cufftExecC2C;
float2 (*make_Scalar2)(float, float) = &make_float2;
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
                  unsigned long long width,
                  unsigned long long height,
                  unsigned long long length,
                  unsigned long long nx,
                  unsigned long long ny,
                  unsigned long long num_fft2,
                  unsigned long long buff_pitch,
                  unsigned long long pitch_nx)
{
    int device_id;
    cudaGetDevice(&device_id);

    // compute half width of fft2
    unsigned long long _nx = nx / 2 + 1;
    // compute batch number of fft2
    unsigned long long batch = (length - 1) / num_fft2 + 1;
    // compute fft2 normalizaton factor
    double norm_fact = 1.0 / sqrt((double)(nx * ny));

    // ***Allocate device arrays
    // workspace
    Scalar *d_workspace;
    gpuErrchk(cudaMalloc(&d_workspace, 2 * pitch_nx * ny * batch * sizeof(Scalar)));
    // buffer (only allocate if T is not double)
    T *d_buff;
    if (!std::is_same<T, Scalar>::value)
    {
        gpuErrchk(cudaMalloc(&d_buff, buff_pitch * height * batch * sizeof(T)));
    }

    // ***Create fft2 plan
    cufftHandle fft2_plan = fft2_create_plan(nx,
                                             ny,
                                             batch,
                                             pitch_nx);

    // Compute efficient execution configuration
    int numSMs; // Number of streaming multiprocessors
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id));
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));

    // copy_convert_kernel
    int blockSize_copy; // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_copy, copy_convert_kernel<T>, 0, 0));
    dim3 gridSize_copy(min(1ULL * maxGridSizeX, batch), min(1ULL * maxGridSizeY, height), 1);

    // scale kernel
    int blockSize_scale; // The launch configurator returned block size
    int gridSize_scale;  // The actual grid size needed, based on input size

    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_scale, scale_array_kernel, 0, 0));
    // Round up according to array size
    gridSize_scale = min((ny * batch + blockSize_scale - 1) / blockSize_scale, 32ULL * numSMs);

    // ***Batched fft2
    for (unsigned long long ii = 0; ii < num_fft2; ii++)
    {
        // rezero workspace array
        gpuErrchk(cudaMemset(d_workspace, 0.0, 2 * pitch_nx * ny * batch * sizeof(Scalar)));

        // ***Copy values to device
        if (std::is_same<T, Scalar>::value)
        {
            // copy values directly to workspace with zero padding
            // number of images to copy
            unsigned long long num_imgs_copy = (ii + 1) * batch > length ? length - ii * batch : batch;
            // use cudaMemcpy3D
            cudaMemcpy3DParms params = {0};
            params.srcArray = NULL;
            params.srcPos = make_cudaPos(0, 0, ii * batch);
            params.srcPtr = make_cudaPitchedPtr((Scalar *)h_in, width * sizeof(Scalar), width, height);
            params.dstArray = NULL;
            params.dstPos = make_cudaPos(0, 0, 0);
            params.dstPtr = make_cudaPitchedPtr(d_workspace, 2 * pitch_nx * sizeof(Scalar), 2 * pitch_nx, ny);
            params.extent = make_cudaExtent(width * sizeof(Scalar), height, num_imgs_copy);
            params.kind = cudaMemcpyHostToDevice;

            gpuErrchk(cudaMemcpy3D(&params));
        }
        else
        {
            // rezero buffer array
            gpuErrchk(cudaMemset(d_buff, (T)0, buff_pitch * height * batch * sizeof(T)));

            // offset index
            unsigned long long offset = ii * width * height * batch;
            // number of rows to copy
            unsigned long long num_rows_copy = (ii + 1) * batch > length ? height * (length - ii * batch) : height * batch;
            // copy values to buffer
            gpuErrchk(cudaMemcpy2D(d_buff, buff_pitch * sizeof(T), h_in + offset, width * sizeof(T), width * sizeof(T), num_rows_copy, cudaMemcpyHostToDevice));

            // convert values of buffer into workspace
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

        // ***Execute fft2 plan
        cufftSafeCall(CufftExecR2C(fft2_plan, d_workspace, (CUFFTCOMPLEX *)d_workspace));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Normalize fft2
        // Starting index
        unsigned long long start = ii * ny * batch;
        // Final index (if exceeds array size, truncate)
        unsigned long long end = (ii + 1) * batch > length ? length * ny : (ii + 1) * ny * batch;
        // scale array
        scale_array_kernel<<<gridSize_scale, blockSize_scale>>>((Scalar2 *)d_workspace,
                                                                pitch_nx,
                                                                _nx,
                                                                norm_fact,
                                                                end - start);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy values back to host
        gpuErrchk(cudaMemcpy2D((Scalar2 *)h_out + _nx * start, _nx * sizeof(Scalar2), (Scalar2 *)d_workspace, pitch_nx * sizeof(Scalar2), _nx * sizeof(Scalar2), end - start, cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace));
    gpuErrchk(cudaFree(d_buff));
    cufftSafeCall(cufftDestroy(fft2_plan));
}

template void compute_fft2<u_int8_t>(const u_int8_t *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<int16_t>(const int16_t *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<u_int16_t>(const u_int16_t *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<int32_t>(const int32_t *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<u_int32_t>(const u_int32_t *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<int64_t>(const int64_t *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<u_int64_t>(const u_int64_t *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<float>(const float *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);
template void compute_fft2<double>(const double *h_in, Scalar *h_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long nx, unsigned long long ny, unsigned long long num_fft2, unsigned long long buff_pitch, unsigned long long pitch_nx);

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
    int device_id;
    cudaGetDevice(&device_id);

    unsigned long long _nx = nx / 2 + 1;                             // fft2 r2c number of complex elements over x
    unsigned long long chunk_size = (_nx * ny - 1) / num_chunks + 1; // number of q points in a chunk

    // ***Allocate space on device
    // workspaces
    Scalar *d_workspace1, *d_workspace2;
    unsigned long long workspace_size = max(pitch_q * length, chunk_size * pitch_t) * 2 * sizeof(Scalar);
    gpuErrchk(cudaMalloc(&d_workspace1, workspace_size));
    gpuErrchk(cudaMalloc(&d_workspace2, workspace_size));
    // helper arrays
    unsigned int *d_lags;
    Scalar2 *d_power_spec, *d_var;
    gpuErrchk(cudaMalloc(&d_lags, lags.size() * sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc(&d_power_spec, chunk_size * sizeof(Scalar2)));
    gpuErrchk(cudaMalloc(&d_var, chunk_size * sizeof(Scalar2)));

    // ***Compute optimal kernels execution parameters
    // transpose_complex_matrix_kernel
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

    // correlation part
    int blockSize_corr = min(nextPowerOfTwo(length), 512ULL);
    int gridSize_corr_x = min(((unsigned long long)(lags.size()) + blockSize_corr - 1) / blockSize_corr, (unsigned long long)maxGridSizeX);
    int gridSize_corr_y = min(chunk_size, (unsigned long long)maxGridSizeY);
    dim3 gridSize_corr(gridSize_corr_x, gridSize_corr_y, 1);
    int smemSize = (blockSize_corr <= 32) ? 2ULL * blockSize_corr * sizeof(double) : 1ULL * blockSize_corr * sizeof(double);

    // power spectrum and variance (reduction)
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int numSMs;      // Number of streaming multiprocessors
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id));

    int blockSize_red = min(nextPowerOfTwo(length), 512ULL);
    int gridSize_red = min(chunk_size, (unsigned long long)maxGridSizeX);
    int smemSize2 = (blockSize_corr <= 32) ? 2ULL * blockSize_corr * sizeof(double2) : 1ULL * blockSize_corr * sizeof(double2);

    // linear combination for variance
    int blockSize_lc; // The launch configurator returned block size
    int gridSize_lc;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_lc, linear_combination_kernel, 0, 0));
    // Round up according to array size
    gridSize_lc = min((chunk_size + blockSize_lc - 1) / blockSize_lc, 32ULL * numSMs);

    // ***Loop over chunks
    for (unsigned long long chunk = 0; chunk < num_chunks; chunk++)
    {
        // ***Get start and end q values
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

        // ***Copy values from host to device
        // elements are complex
        // to speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D(d_workspace2,
                               2 * pitch_q * sizeof(Scalar),
                               h_in + 2 * q_start,
                               2 * _nx * ny * sizeof(Scalar),
                               2 * curr_chunk_size * sizeof(Scalar),
                               length,
                               cudaMemcpyHostToDevice));

        // ***Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran1, blockSize_tran>>>((Scalar2 *)d_workspace2,
                                                                            pitch_q,
                                                                            (Scalar2 *)d_workspace1,
                                                                            pitch_t,
                                                                            curr_chunk_size,
                                                                            length,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM,
                                                                            (length + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Zero-out workspace2
        gpuErrchk(cudaMemset(d_workspace2, 0.0, workspace_size));

        // ***Compute structure function using differences (d_workspace1 --> d_workspace2)
        structure_function_diff_kernel<<<gridSize_corr, blockSize_corr, smemSize>>>((Scalar2 *)d_workspace1,
                                                                                    (Scalar2 *)d_workspace2,
                                                                                    d_lags,
                                                                                    length,
                                                                                    lags.size(),
                                                                                    curr_chunk_size,
                                                                                    pitch_t);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Compute power spectrum (d_workspace1 --> d_power_spect)
        average_power_spectrum_kernel<<<gridSize_red, blockSize_red, smemSize>>>((Scalar2 *)d_workspace1,
                                                                                 d_power_spec,
                                                                                 length,
                                                                                 pitch_t,
                                                                                 curr_chunk_size);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Compute variance (d_workspace1 --> d_var)
        // compute average over time
        average_complex_kernel<<<gridSize_red, blockSize_red, smemSize2>>>((Scalar2 *)d_workspace1,
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

        // ***Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran2, blockSize_tran>>>((Scalar2 *)d_workspace2,
                                                                            pitch_t,
                                                                            (Scalar2 *)d_workspace1,
                                                                            pitch_q,
                                                                            lags.size(),
                                                                            curr_chunk_size,
                                                                            (lags.size() + TILE_DIM - 1) / TILE_DIM,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy values from device to host
        // elements are treated as complex
        // to speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D(h_in + 2 * q_start,
                               2 * _nx * ny * sizeof(Scalar),
                               d_workspace1,
                               2 * pitch_q * sizeof(Scalar),
                               2 * curr_chunk_size * sizeof(Scalar),
                               lags.size(),
                               cudaMemcpyDeviceToHost));
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
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
    gpuErrchk(cudaFree(d_lags));
    gpuErrchk(cudaFree(d_power_spec));
    gpuErrchk(cudaFree(d_var));
}

/*! \brief Convert to full and fftshifted image structure function on the GPU
    \param h_in             input array after structure function calculation
    \param Nlags            number of lags analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param num_fullshift    number of full and shift chunks
    \param pitch_fs         pitch of device array for full and shift operations
 */
void make_full_shift(Scalar *h_in,
                     unsigned long long Nlags,
                     unsigned long long nx,
                     unsigned long long ny,
                     unsigned long long num_fullshift,
                     unsigned long long pitch_fs)
{
    int device_id;
    cudaGetDevice(&device_id);

    unsigned long long _nx = nx / 2 + 1;                             // fft2 r2c number of complex elements over x
    unsigned long long chunk_size = (Nlags - 1) / num_fullshift + 1; // number of lags in a chunk

    // ***Allocate space on device
    // workspaces
    Scalar *d_workspace1, *d_workspace2;
    gpuErrchk(cudaMalloc(&d_workspace1, pitch_fs * ny * chunk_size * 2 * sizeof(Scalar)));
    gpuErrchk(cudaMalloc(&d_workspace2, pitch_fs * ny * chunk_size * 2 * sizeof(Scalar)));

    // ***Compute optimal kernels execution parameters
    dim3 blockSize_full(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, device_id));
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, device_id));
    int gridSize_full_x = min((_nx + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_full_y = min((ny * chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_full(gridSize_full_x, gridSize_full_y, 1);
    int gridSize_shift_x = min((nx + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeX);
    int gridSize_shift_y = min((ny * chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
    dim3 gridSize_shift(gridSize_shift_x, gridSize_shift_y, 1);

    // ***Loop over chunks
    for (unsigned long long chunk = 0; chunk < num_fullshift; chunk++)
    {
        // Get input offset
        unsigned long long ioffset = chunk * chunk_size * 2 * _nx * ny;
        // Get current chunk size
        unsigned long long curr_chunk_size = (chunk + 1) * chunk_size > Nlags ? Nlags - chunk * chunk_size : chunk_size;

        if (curr_chunk_size != chunk_size)
        {
            gridSize_full_y = min((ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
            gridSize_shift_y = min((ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (unsigned long long)maxGridSizeY);
            gridSize_full.y = gridSize_full_y;
            gridSize_shift.y = gridSize_shift_y;
        }

        // ***Copy values from host to device
        gpuErrchk(cudaMemcpy2D(d_workspace1,
                               pitch_fs * 2 * sizeof(Scalar),
                               h_in + ioffset,
                               2 * _nx * sizeof(Scalar),
                               2 * _nx * sizeof(Scalar),
                               curr_chunk_size * ny,
                               cudaMemcpyHostToDevice));

        // ***Make full power spectrum (workspace1 --> workspace2)
        make_full_powspec_kernel<<<gridSize_full, blockSize_full>>>((Scalar2 *)d_workspace1,
                                                                    pitch_fs,
                                                                    d_workspace2,
                                                                    2 * pitch_fs,
                                                                    _nx,
                                                                    nx,
                                                                    ny,
                                                                    curr_chunk_size,
                                                                    (_nx + TILE_DIM - 1) / TILE_DIM,
                                                                    (ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Shift power spectrum (workspace2 --> workspace1)
        shift_powspec_kernel<<<gridSize_shift, blockSize_full>>>(d_workspace2,
                                                                 2 * pitch_fs,
                                                                 d_workspace1,
                                                                 2 * pitch_fs,
                                                                 nx,
                                                                 ny,
                                                                 curr_chunk_size,
                                                                 (nx + TILE_DIM - 1) / TILE_DIM,
                                                                 (ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy values from device to host (make contiguous on memory)
        // Get output offset
        unsigned long long ooffset = chunk * chunk_size * nx * ny;
        gpuErrchk(cudaMemcpy2D(h_in + ooffset,
                               nx * sizeof(Scalar),
                               d_workspace1,
                               pitch_fs * 2 * sizeof(Scalar),
                               nx * sizeof(Scalar),
                               curr_chunk_size * ny,
                               cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
}

/*!
    Compute image structure function using the WK theorem on the GPU
 */
void structure_function_fft(Scalar *h_in,
                            vector<unsigned int> lags,
                            unsigned long long length,
                            unsigned long long nx,
                            unsigned long long ny,
                            unsigned long long nt,
                            unsigned long long num_chunks,
                            unsigned long long pitch_q,
                            unsigned long long pitch_t,
                            unsigned long long pitch_nt)
{
    int device_id;
    cudaGetDevice(&device_id);

    unsigned long long _nx = nx / 2 + 1;                             // fft2 r2c number of complex elements over x
    unsigned long long chunk_size = (_nx * ny - 1) / num_chunks + 1; // number of q points in a chunk

    // ***Allocate memory on device for fft
    Scalar *d_workspace1, *d_workspace2;
    // To speed up memory trasfer and meet the alignment requirements for coalescing,
    // we space each subarray by pitch1 (over q) or pitch2 (over t)
    gpuErrchk(cudaMalloc(&d_workspace1, chunk_size * pitch_nt * 2 * sizeof(Scalar)));
    unsigned long long workspace2_size = max(chunk_size * pitch_t, pitch_q * length);
    gpuErrchk(cudaMalloc(&d_workspace2, workspace2_size * 2 * sizeof(Scalar)));

    // helper arrays
    unsigned int *d_lags;
    gpuErrchk(cudaMalloc(&d_lags, lags.size() * sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    Scalar2 *d_power_spec, *d_var;
    gpuErrchk(cudaMalloc(&d_power_spec, chunk_size * sizeof(Scalar2)));
    gpuErrchk(cudaMalloc(&d_var, chunk_size * sizeof(Scalar2)));

    // ***Create fft plan
    cufftHandle fft_plan = fft_create_plan(nt,
                                           chunk_size,
                                           pitch_nt);

    // ***Compute efficient kernels execution configurations
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int numSMs;      // Number of streaming multiprocessors
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device_id));

    // transpose_complex_matrix_kernel
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

    // square_modulus_kernel
    int blockSize_sqmod; // The launch configurator returned block size
    int gridSize_sqmod;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_sqmod, square_modulus_kernel, 0, 0));
    // Round up according to array size
    gridSize_sqmod = min((chunk_size + blockSize_sqmod - 1) / blockSize_sqmod, 32ULL * numSMs);

    // scale_array_kernel
    int blockSize_scale; // The launch configurator returned block size
    int gridSize_scale;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_scale, scale_array_kernel, 0, 0));
    // Round up according to array size
    gridSize_scale = min((chunk_size + blockSize_scale - 1) / blockSize_scale, 32ULL * numSMs);

    // linear_combination_final_kernel
    int blockSize_final; // The launch configurator returned block size
    int gridSize_final;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_final, linear_combination_final_kernel, 0, 0));
    // Round up according to array size
    gridSize_final = min(chunk_size, 32ULL * numSMs);

    // copy_selected_lags_kernel
    int blockSize_copy; // The launch configurator returned block size
    int gridSize_copy;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_copy, copy_selected_lags_kernel, 0, 0));
    // Round up according to array size
    gridSize_copy = min(chunk_size, 32ULL * numSMs);

    // linear combination for power spectrum and variance
    int blockSize_lc; // The launch configurator returned block size
    int gridSize_lc;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_lc, linear_combination_kernel, 0, 0));
    // Round up according to array size
    gridSize_lc = min((chunk_size + blockSize_lc - 1) / blockSize_lc, 32ULL * numSMs);

    // ***Loop over the chunks
    for (unsigned long long chunk = 0; chunk < num_chunks; chunk++)
    {
        // ***Get initial and final q indices
        unsigned long long q_start = chunk * chunk_size;
        unsigned long long q_end = (chunk + 1) * chunk_size < _nx * ny ? (chunk + 1) * chunk_size : _nx * ny;
        unsigned long long curr_chunk_size = q_end - q_start;

        // ***Check if fft plans are still ok (batch number must be the same as previous one)
        // This may be true during the very last iteration only
        if (curr_chunk_size != chunk_size)
        {
            cufftSafeCall(cufftDestroy(fft_plan));
            fft_plan = fft_create_plan(nt,
                                       curr_chunk_size,
                                       pitch_nt);
            // also change gridSize for transpose and linear combination
            gridSize_tran1.x = (curr_chunk_size + TILE_DIM - 1) / TILE_DIM;
            gridSize_tran2.y = (curr_chunk_size + TILE_DIM - 1) / TILE_DIM;
            gridSize_final = min(curr_chunk_size, 32ULL * numSMs);
            gridSize_sqmod = min((curr_chunk_size + blockSize_sqmod - 1) / blockSize_sqmod, 32ULL * numSMs);
            gridSize_scale = min((curr_chunk_size + blockSize_scale - 1) / blockSize_scale, 32ULL * numSMs);
        }

        // ***Zero-out elements of workspace1 and workspace2 device arrays
        gpuErrchk(cudaMemset2D(d_workspace1, 2 * pitch_nt * sizeof(Scalar), 0.0, 2 * nt * sizeof(Scalar), curr_chunk_size));
        gpuErrchk(cudaMemset(d_workspace2, 0.0, 2 * workspace2_size * sizeof(Scalar)));

        // ***Copy values from host buffer to device workspace2 with pitch_q
        unsigned long long offset = 2 * q_start;                     // host source array offset
        unsigned long long spitch = 2 * (_nx * ny) * sizeof(Scalar); // host source array pitch
        unsigned long long dpitch = 2 * pitch_q * sizeof(Scalar);    // device destination array pitch
        gpuErrchk(cudaMemcpy2D(d_workspace2,
                               dpitch,
                               h_in + offset,
                               spitch,
                               2 * curr_chunk_size * sizeof(Scalar),
                               length,
                               cudaMemcpyHostToDevice));

        // ***Transpose complex matrix ({d_workspace2; pitch_q} --> {d_workspace1; pitch_nt})
        transpose_complex_matrix_kernel<<<gridSize_tran1, blockSize_tran>>>((Scalar2 *)d_workspace2,
                                                                            pitch_q,
                                                                            (Scalar2 *)d_workspace1,
                                                                            pitch_nt,
                                                                            curr_chunk_size,
                                                                            length,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM,
                                                                            (length + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy values ({d_workspace1; pitch_nt} --> {d_workspace2; pitch_t})
        dpitch = 2 * pitch_t * sizeof(Scalar);
        spitch = 2 * pitch_nt * sizeof(Scalar);
        gpuErrchk(cudaMemcpy2D(d_workspace2,
                               dpitch,
                               d_workspace1,
                               spitch,
                               length * 2 * sizeof(Scalar),
                               curr_chunk_size,
                               cudaMemcpyDeviceToDevice));

        // +++ FFT PART +++
        // ***Do fft (d_workspace1 --> d_workspace1)
        cufftSafeCall(CufftExecC2C(fft_plan, (CUFFTCOMPLEX *)d_workspace1, (CUFFTCOMPLEX *)d_workspace1, CUFFT_FORWARD));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Compute square modulus (d_workspace1 --> d_workspace1)
        square_modulus_kernel<<<gridSize_sqmod, blockSize_sqmod>>>((Scalar2 *)d_workspace1,
                                                                   nt,
                                                                   pitch_nt,
                                                                   curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy square modulus of sum part
        gpuErrchk(cudaMemcpy2D(d_var,
                               sizeof(Scalar2),
                               d_workspace1,
                               pitch_nt * sizeof(Scalar2),
                               sizeof(Scalar2),
                               curr_chunk_size,
                               cudaMemcpyDeviceToDevice));

        // ***Do fft (d_workspace1 --> d_workspace1)
        cufftSafeCall(CufftExecC2C(fft_plan, (CUFFTCOMPLEX *)d_workspace1, (CUFFTCOMPLEX *)d_workspace1, CUFFT_FORWARD));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Scale fft part (d_workspace1 --> d_workspace1)
        scale_array_kernel<<<gridSize_scale, blockSize_scale>>>((Scalar2 *)d_workspace1,
                                                                pitch_nt,
                                                                nt,
                                                                1.0 / (double)nt,
                                                                curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy power spectrum part
        gpuErrchk(cudaMemcpy2D(d_power_spec,
                               sizeof(Scalar2),
                               d_workspace1,
                               pitch_nt * sizeof(Scalar2),
                               sizeof(Scalar2),
                               curr_chunk_size,
                               cudaMemcpyDeviceToDevice));

        // scale power spectrum (d_power_spec = d_power_spec * 1/length + 0)
        linear_combination_kernel<<<gridSize_lc, blockSize_lc>>>(d_power_spec,
                                                                 d_power_spec,
                                                                 make_double2(1.0 / (double)length, 0.0),
                                                                 d_power_spec,
                                                                 make_double2(0.0, 0.0),
                                                                 curr_chunk_size);

        // ***Compute variance (d_var = (1.0, 0.0) * d_power_spec + (-1.0/(length * length), 0.0) * d_var)
        linear_combination_kernel<<<gridSize_lc, blockSize_lc>>>(d_var,
                                                                 d_power_spec,
                                                                 make_double2(1.0, 0.0),
                                                                 d_var,
                                                                 make_double2(-1.0 / (double)(length * length), 0.0),
                                                                 curr_chunk_size);

        // +++ CUMULATIVE SUM PART +++
        // ***Compute square modulus (d_workspace2 --> d_workspace2)
        square_modulus_kernel<<<gridSize_sqmod, blockSize_sqmod>>>((Scalar2 *)d_workspace2,
                                                                   length,
                                                                   pitch_t,
                                                                   curr_chunk_size);

        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy the value into the imaginary part of the opposite (with respect to time) element (d_workspace2 --> d_workspace2)
        real2imagopposite_kernel<<<gridSize_sqmod, blockSize_sqmod>>>((Scalar2 *)d_workspace2,
                                                                      length,
                                                                      pitch_t,
                                                                      curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Compute (exclusive) cumulative sum (prefix scan)
        scan_wrap(d_workspace2,
                  d_workspace2,
                  2 * length,
                  2 * pitch_t,
                  curr_chunk_size);

        //  ***Linearly combine the two parts (workspace1 + workspace2 --> workspace2)
        linear_combination_final_kernel<<<gridSize_final, blockSize_final>>>((Scalar2 *)d_workspace2,
                                                                             pitch_t,
                                                                             (Scalar2 *)d_workspace1,
                                                                             pitch_nt,
                                                                             (Scalar2 *)d_workspace2,
                                                                             pitch_t,
                                                                             length,
                                                                             curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Keep only selected lags ({d_workspace2; pitch_t} --> {d_workspace1; pitch_t})
        copy_selected_lags_kernel<<<gridSize_copy, blockSize_copy>>>((Scalar2 *)d_workspace2,
                                                                     (Scalar2 *)d_workspace1,
                                                                     d_lags,
                                                                     lags.size(),
                                                                     pitch_t,
                                                                     pitch_t,
                                                                     curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Transpose complex matrix ({d_workspace1; pitch_t} --> {d_workspace2; pitch_q})
        transpose_complex_matrix_kernel<<<gridSize_tran2, blockSize_tran>>>((Scalar2 *)d_workspace1,
                                                                            pitch_t,
                                                                            (Scalar2 *)d_workspace2,
                                                                            pitch_q,
                                                                            lags.size(),
                                                                            curr_chunk_size,
                                                                            (lags.size() + TILE_DIM - 1) / TILE_DIM,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy from device to host (d_workspace2 --> host)
        // elements are treated as complex
        // to speed up transfer, use pitch_q
        offset = 2 * q_start;                     // host destination array offset
        spitch = 2 * pitch_q * sizeof(Scalar);    // host source array pitch
        dpitch = 2 * (_nx * ny) * sizeof(Scalar); // device destination array pitch
        gpuErrchk(cudaMemcpy2D(h_in + offset,
                               dpitch,
                               d_workspace2,
                               spitch,
                               2 * curr_chunk_size * sizeof(Scalar),
                               lags.size(),
                               cudaMemcpyDeviceToHost));

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
    }

    // ***Free memory
    cufftSafeCall(cufftDestroy(fft_plan));
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
    gpuErrchk(cudaFree(d_lags));
    gpuErrchk(cudaFree(d_power_spec));
    gpuErrchk(cudaFree(d_var));
}
