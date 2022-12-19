// Maintainer: enrico-lattuada

/*! \file dfm_cuda.cu
    \brief Definition of core CUDA Digital Fourier Microscopy functions
*/

// *** headers ***
#include "dfm_cuda.cuh"

#include "helper_debug.cuh"
#include "helper_cufft.cuh"
#include "helper_dfm_cuda.cuh"
#include "helper_prefix_sum.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdlib.h>

// #include <chrono>
// using namespace std::chrono;

#define CUFFTCOMPLEX cufftDoubleComplex

// *** code ***
const unsigned int TILE_DIM = 32;  // leave this unchanged!! (tile dimension for matrix transpose)
const unsigned int BLOCK_ROWS = 8; // leave this unchanged!! (block rows for matrix transpose)

/*!
    Evaluate the device memory pitch for multiple subarrays of size N with 8bytes elements
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
    Evaluate the device memory size in bytes for fft2
*/
void cudaGetFft2MemSize(size_t nx,
                        size_t ny,
                        size_t batch,
                        size_t *memsize)
{
    fft2_get_mem_size(nx,
                      ny,
                      batch,
                      memsize);
}

/*!
    Evaluate the device memory size in bytes for fft
*/
void cudaGetFftMemSize(size_t nt,
                       size_t batch,
                       size_t pitch,
                       size_t *memsize)
{
    fft_get_mem_size(nt,
                     batch,
                     pitch,
                     memsize);
}

/*!
    Transfer images on GPU and compute fft2
 */
template <typename T>
void compute_fft2(const T *h_in,
                  double *h_out,
                  size_t width,
                  size_t height,
                  size_t length,
                  size_t nx,
                  size_t ny,
                  size_t num_fft2,
                  size_t buff_pitch)
{
    // compute half width of fft2
    size_t _nx = nx / 2 + 1;
    // compute batch number of fft2
    size_t batch = (length - 1) / num_fft2 + 1;
    // compute fft2 normalizaton factor
    double norm_fact = 1.0 / sqrt((double)(nx * ny));

    // ***Allocate device arrays
    // workspace
    double *d_workspace;
    gpuErrchk(cudaMalloc(&d_workspace, 2 * _nx * ny * batch * sizeof(double)));
    // buffer (only allocate if T is not double)
    T *d_buff;
    if (!std::is_same<T, double>::value)
    {
        gpuErrchk(cudaMalloc(&d_buff, buff_pitch * height * batch * sizeof(T)));
    }

    // ***Create fft2 plan
    cufftHandle fft2_plan = fft2_create_plan(nx,
                                             ny,
                                             batch);

    // Compute efficient execution configuration
    // copy/convert kernel
    int blockSize_copy = 512;                                                           // The launch configurator returned block size
    int gridSize_copy = (width * height * batch + blockSize_copy - 1) / blockSize_copy; // The actual grid size needed, based on input size

    // scale kernel
    int blockSize_scale; // The launch configurator returned block size
    int minGridSize;     // The minimum grid size needed to achieve the
                         // maximum occupancy for a full device launch
    int gridSize_scale;  // The actual grid size needed, based on input size

    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_scale, scale_array_kernel, 0, 0));
    // Round up according to array size
    gridSize_scale = (2 * _nx * ny * batch + blockSize_scale - 1) / blockSize_scale;

    // ***Batched fft2
    for (size_t ii = 0; ii < num_fft2; ii++)
    {
        // rezero workspace array
        gpuErrchk(cudaMemset(d_workspace, 0.0, 2 * _nx * ny * batch * sizeof(double)));

        // ***Copy values to device
        if (std::is_same<T, double>::value)
        {
            // copy values directly to workspace with zero padding
            // number of images to copy
            size_t num_imgs_copy = (ii + 1) * batch > length ? length - ii * batch : batch;
            // use cudaMemcpy3D
            cudaMemcpy3DParms params = {0};
            params.srcArray = NULL;
            params.srcPos = make_cudaPos(0, 0, ii * batch);
            params.srcPtr = make_cudaPitchedPtr((double *)h_in, width * sizeof(double), width, height);
            params.dstArray = NULL;
            params.dstPos = make_cudaPos(0, 0, 0);
            params.dstPtr = make_cudaPitchedPtr(d_workspace, 2 * _nx * sizeof(double), 2 * _nx, ny);
            params.extent = make_cudaExtent(width * sizeof(double), height, num_imgs_copy);
            params.kind = cudaMemcpyHostToDevice;

            gpuErrchk(cudaMemcpy3D(&params));
        }
        else
        {
            // rezero buffer array
            gpuErrchk(cudaMemset(d_buff, (T)0, buff_pitch * height * batch * sizeof(T)));

            // offset index
            size_t offset = ii * width * height * batch;
            // number of rows to copy
            size_t num_rows_copy = (ii + 1) * batch > length ? height * (length - ii * batch) : height * batch;
            // copy values to buffer
            gpuErrchk(cudaMemcpy2D(d_buff, buff_pitch * sizeof(T), h_in + offset, width * sizeof(T), width * sizeof(T), num_rows_copy, cudaMemcpyHostToDevice));

            // convert values of buffer into workspace
            copy_convert_kernel<<<gridSize_copy, blockSize_copy>>>(d_buff,
                                                                   d_workspace,
                                                                   width,
                                                                   width * height,
                                                                   buff_pitch,
                                                                   buff_pitch * height,
                                                                   2 * _nx,
                                                                   2 * _nx * ny,
                                                                   width * height * batch);
        }

        // ***Execute fft2 plan
        cufftSafeCall(cufftExecD2Z(fft2_plan, d_workspace, (CUFFTCOMPLEX *)d_workspace));

        // ***Normalize fft2
        // Starting index
        size_t start = 2 * ii * _nx * ny * batch;
        // Final index (if exceeds array size, truncate)
        size_t end = (ii + 1) * batch > length ? 2 * length * _nx * ny : 2 * (ii + 1) * _nx * ny * batch;
        // scale array
        scale_array_kernel<<<gridSize_scale, blockSize_scale>>>(d_workspace,
                                                                norm_fact,
                                                                d_workspace,
                                                                end - start);

        // ***Copy values back to host
        gpuErrchk(cudaMemcpy(h_out + start, d_workspace, (end - start) * sizeof(double), cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace));
    gpuErrchk(cudaFree(d_buff));
    cufftSafeCall(cufftDestroy(fft2_plan));
}

template void compute_fft2<double>(const double *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<float>(const float *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<int64_t>(const int64_t *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<int32_t>(const int32_t *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<int16_t>(const int16_t *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<u_int64_t>(const u_int64_t *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<u_int32_t>(const u_int32_t *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<u_int16_t>(const u_int16_t *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);
template void compute_fft2<u_int8_t>(const u_int8_t *h_in, double *h_out, size_t width, size_t height, size_t length, size_t nx, size_t ny, size_t num_fft2, size_t buff_pitch);

/*!
    Compute Image Structure Factor using differences on the GPU
 */
void correlate_direct(double *h_in,
                      vector<unsigned int> lags,
                      size_t length,
                      size_t nx,
                      size_t ny,
                      size_t num_chunks,
                      size_t pitch_q,
                      size_t pitch_t)
{
    size_t _nx = nx / 2 + 1;                             // fft2 r2c number of complex elements over x
    size_t chunk_size = (_nx * ny - 1) / num_chunks + 1; // number of q points in a chunk

    // ***Create vector of t1 and num
    // The maximum size of t1 is (length-min_lag)*num_lags
    vector<unsigned int> t1((length - lags[0]) * lags.size());
    vector<unsigned int> num(length - lags[0]);
    unsigned int N = 0;
    for (unsigned int t = 0; t < length - lags[0]; t++)
    {
        num[t] = N;
        for (unsigned int dt : lags)
        {
            if (t + dt < length)
            {
                t1[N] = t;
                N++;
            }
            else
            {
                break;
            }
        }
    }

    // ***Allocate space on device
    // workspaces
    double *d_workspace1, *d_workspace2;
    size_t workspace_size = max(pitch_q * length, chunk_size * pitch_t) * 2 * sizeof(double);
    gpuErrchk(cudaMalloc(&d_workspace1, workspace_size));
    gpuErrchk(cudaMalloc(&d_workspace2, workspace_size));
    // helper arrays
    unsigned int *d_t1, *d_lags, *d_num;
    gpuErrchk(cudaMalloc(&d_lags, lags.size() * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_t1, N * sizeof(unsigned int)));
    gpuErrchk(cudaMalloc(&d_num, num.size() * sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_t1, t1.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_num, num.data(), num.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    t1.clear();
    t1.shrink_to_fit();
    num.clear();
    num.shrink_to_fit();

    // ***Compute optimal kernels execution parameters
    // transpose_complex_matrix_kernel
    dim3 blockSize_tran(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, 0)); // gpu id fixed to 0 for now
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, 0)); // gpu id fixed to 0 for now
    int gridSize_tran1_x = min((chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeX);
    int gridSize_tran1_y = min((length + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
    dim3 gridSize_tran1(gridSize_tran1_x, gridSize_tran1_y, 1);
    int gridSize_tran2_x = min((lags.size() + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeX);
    int gridSize_tran2_y = min((chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
    dim3 gridSize_tran2(gridSize_tran2_x, gridSize_tran2_y, 1);

    // correlation part
    int blockSize_corr; // The launch configurator returned block size
    int minGridSize;    // The minimum grid size needed to achieve the
                        // maximum occupancy for a full device launch
    int gridSize_corr;  // The actual grid size needed, based on input size
    int numSMs;
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0)); // gpu id fixed to 0 for now

    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_corr, correlatewithdifferences_kernel, 0, 0));
    // Round up according to array size
    gridSize_corr = min((chunk_size * N + blockSize_corr - 1) / blockSize_corr, 32 * (size_t)numSMs);

    // ***Loop over chunks
    for (size_t chunk = 0; chunk < num_chunks; chunk++)
    {
        // ***Get start and end q values
        size_t q_start = chunk * chunk_size;
        size_t q_end = (chunk + 1) * chunk_size < _nx * ny ? (chunk + 1) * chunk_size : _nx * ny;
        size_t curr_chunk_size = q_end - q_start;

        if (curr_chunk_size != chunk_size)
        {
            // if chunk size changes, modify kernel execution parameters for transpose
            gridSize_tran1.x = min((curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeX);
            gridSize_tran2.y = min((curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
        }

        // ***Copy values from host to device
        // elements are complex doubles
        // to speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D(d_workspace2,
                               2 * pitch_q * sizeof(double),
                               h_in + 2 * q_start,
                               2 * _nx * ny * sizeof(double),
                               2 * curr_chunk_size * sizeof(double),
                               length,
                               cudaMemcpyHostToDevice));

        // ***Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran1, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_q,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_t,
                                                                            curr_chunk_size,
                                                                            length,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM,
                                                                            (length + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());

        // ***Zero-out workspace2
        gpuErrchk(cudaMemset(d_workspace2, 0.0, workspace_size));

        // ***Correlate using differences (d_workspace1 --> d_workspace2)
        correlatewithdifferences_kernel<<<gridSize_corr, blockSize_corr>>>((double2 *)d_workspace1,
                                                                           (double2 *)d_workspace2,
                                                                           d_lags,
                                                                           d_t1,
                                                                           d_num,
                                                                           length,
                                                                           lags.size(),
                                                                           curr_chunk_size,
                                                                           N,
                                                                           pitch_t);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Transpose array (d_workspace2 --> d_workspace1)
        transpose_complex_matrix_kernel<<<gridSize_tran2, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_t,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_q,
                                                                            lags.size(),
                                                                            curr_chunk_size,
                                                                            (lags.size() + TILE_DIM - 1) / TILE_DIM,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());

        // ***Copy values from device to host
        // elements are treated as complex doubles
        // to speed up transfer, use pitch_q
        gpuErrchk(cudaMemcpy2D(h_in + 2 * q_start,
                               2 * _nx * ny * sizeof(double),
                               d_workspace1,
                               2 * pitch_q * sizeof(double),
                               2 * curr_chunk_size * sizeof(double),
                               lags.size(),
                               cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
    gpuErrchk(cudaFree(d_lags));
    gpuErrchk(cudaFree(d_t1));
    gpuErrchk(cudaFree(d_num));
}

/*! \brief Convert to full and fftshifted Image Structure Function on the GPU
    \param h_in             input array after structure function calculation
    \param lags             lags to be analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param num_fullshift    number of full and shift chunks
    \param pitch_fs         pitch of device array for full and shift operations
 */
void make_full_shift(double *h_in,
                     vector<unsigned int> lags,
                     size_t nx,
                     size_t ny,
                     size_t num_fullshift,
                     size_t pitch_fs)
{
    size_t _nx = nx / 2 + 1;                                   // fft2 r2c number of complex elements over x
    size_t chunk_size = (lags.size() - 1) / num_fullshift + 1; // number of lags in a chunk

    // ***Allocate space on device
    // workspaces
    double *d_workspace1, *d_workspace2;
    gpuErrchk(cudaMalloc(&d_workspace1, pitch_fs * ny * chunk_size * 2 * sizeof(double)));
    gpuErrchk(cudaMalloc(&d_workspace2, pitch_fs * ny * chunk_size * 2 * sizeof(double)));

    // ***Compute optimal kernels execution parameters
    dim3 blockSize_full(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, 0)); // gpu id fixed to 0 for now
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, 0)); // gpu id fixed to 0 for now
    int gridSize_full_x = min((_nx + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeX);
    int gridSize_full_y = min((ny * chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
    dim3 gridSize_full(gridSize_full_x, gridSize_full_y, 1);
    int gridSize_shift_x = min((nx + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeX);
    int gridSize_shift_y = min((ny * chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
    dim3 gridSize_shift(gridSize_shift_x, gridSize_shift_y, 1);

    // ***Loop over chunks
    for (size_t chunk = 0; chunk < num_fullshift; chunk++)
    {
        // Get input offset
        size_t ioffset = chunk * chunk_size * 2 * _nx * ny;
        // Get current chunk size
        size_t curr_chunk_size = (chunk + 1) * chunk_size > lags.size() ? lags.size() - chunk * chunk_size : chunk_size;

        if (curr_chunk_size != chunk_size)
        {
            gridSize_full_y = min((ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
            gridSize_shift_y = min((ny * curr_chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
            gridSize_full.y = gridSize_full_y;
            gridSize_shift.y = gridSize_shift_y;
        }

        // ***Copy values from host to device
        gpuErrchk(cudaMemcpy2D(d_workspace1,
                               pitch_fs * 2 * sizeof(double),
                               h_in + ioffset,
                               2 * _nx * sizeof(double),
                               2 * _nx * sizeof(double),
                               curr_chunk_size * ny,
                               cudaMemcpyHostToDevice));

        // ***Make full power spectrum (workspace1 --> workspace2)
        make_full_powspec_kernel<<<gridSize_full, blockSize_full>>>((double2 *)d_workspace1,
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

        // ***Copy values from device to host (make contiguous on memory)
        // Get output offset
        size_t ooffset = chunk * chunk_size * nx * ny;
        gpuErrchk(cudaMemcpy2D(h_in + ooffset,
                               nx * sizeof(double),
                               d_workspace1,
                               pitch_fs * 2 * sizeof(double),
                               nx * sizeof(double),
                               curr_chunk_size * ny,
                               cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
}

/*!
    Compute Image Structure Factor using the WK theorem on the GPU
 */
void correlate_fft(double *h_in,
                   vector<unsigned int> lags,
                   size_t length,
                   size_t nx,
                   size_t ny,
                   size_t nt,
                   size_t num_chunks,
                   size_t pitch_q,
                   size_t pitch_t,
                   size_t pitch_nt)
{
    size_t _nx = nx / 2 + 1;                             // fft2 r2c number of complex elements over x
    size_t chunk_size = (_nx * ny - 1) / num_chunks + 1; // number of q points in a chunk

    // ***Allocate memory on device for fft
    double *d_workspace1, *d_workspace2;
    // To speed up memory trasfer and meet the alignment requirements for coalescing,
    // we space each subarray by pitch1 (over q) or pitch2 (over t)
    gpuErrchk(cudaMalloc(&d_workspace1, chunk_size * pitch_nt * 2 * sizeof(double)));
    size_t workspace2_size = max(chunk_size * pitch_t, pitch_q * length);
    gpuErrchk(cudaMalloc(&d_workspace2, workspace2_size * 2 * sizeof(double)));

    // helper arrays
    unsigned int *d_lags;
    gpuErrchk(cudaMalloc(&d_lags, lags.size() * sizeof(unsigned int)));
    gpuErrchk(cudaMemcpy(d_lags, lags.data(), lags.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // ***Create fft plan
    cufftHandle fft_plan = fft_create_plan(nt,
                                           chunk_size,
                                           pitch_nt);

    // ***Compute efficient kernels execution configurations
    int minGridSize; // The minimum grid size needed to achieve the
                     // maximum occupancy for a full device launch
    int numSMs;      // Number of streaming multiprocessors
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));

    // transpose_complex_matrix_kernel
    dim3 blockSize_tran(TILE_DIM, BLOCK_ROWS, 1);
    int maxGridSizeX, maxGridSizeY;
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeX, cudaDevAttrMaxGridDimX, 0)); // gpu id fixed to 0 for now
    gpuErrchk(cudaDeviceGetAttribute(&maxGridSizeY, cudaDevAttrMaxGridDimY, 0)); // gpu id fixed to 0 for now
    int gridSize_tran1_x = min((chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeX);
    int gridSize_tran1_y = min((length + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
    dim3 gridSize_tran1(gridSize_tran1_x, gridSize_tran1_y, 1);
    int gridSize_tran2_x = min((lags.size() + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeX);
    int gridSize_tran2_y = min((chunk_size + TILE_DIM - 1) / TILE_DIM, (size_t)maxGridSizeY);
    dim3 gridSize_tran2(gridSize_tran2_x, gridSize_tran2_y, 1);

    // square_modulus_kernel
    int blockSize_sqmod; // The launch configurator returned block size
    int gridSize_sqmod1; // The actual grid size needed, based on input size
    int gridSize_sqmod2;
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_sqmod, square_modulus_kernel, 0, 0));
    // Round up according to array size
    gridSize_sqmod1 = min((chunk_size * pitch_nt / 2 + blockSize_sqmod - 1) / blockSize_sqmod, 32 * (size_t)numSMs);
    gridSize_sqmod2 = min((chunk_size * pitch_t / 2 + blockSize_sqmod - 1) / blockSize_sqmod, 32 * (size_t)numSMs);

    // scale_array_kernel
    int blockSize_scale; // The launch configurator returned block size
    int gridSize_scale;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_scale, scale_array_kernel, 0, 0));
    // Round up according to array size
    gridSize_scale = min((2 * chunk_size * pitch_nt + blockSize_scale - 1) / blockSize_scale, 32 * (size_t)numSMs);

    // linear_combination_final_kernel
    int blockSize_final; // The launch configurator returned block size
    int gridSize_final;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_final, linear_combination_final_kernel, 0, 0));
    // Round up according to array size
    gridSize_final = min(chunk_size, 32 * (size_t)numSMs);

    // copy_selected_lags_kernel
    int blockSize_copy; // The launch configurator returned block size
    int gridSize_copy;  // The actual grid size needed, based on input size
    gpuErrchk(cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize_copy, copy_selected_lags_kernel, 0, 0));
    // Round up according to array size
    gridSize_copy = min(chunk_size, 32 * (size_t)numSMs);

    // ***Loop over the chunks
    for (size_t chunk = 0; chunk < num_chunks; chunk++)
    {
        // ***Get initial and final q indices
        size_t q_start = chunk * chunk_size;
        size_t q_end = (chunk + 1) * chunk_size < _nx * ny ? (chunk + 1) * chunk_size : _nx * ny;
        size_t curr_chunk_size = q_end - q_start;

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
            gridSize_final = min(curr_chunk_size, 32 * (size_t)numSMs);
        }

        // ***Zero-out elements of workspace1 and workspace2 device arrays
        gpuErrchk(cudaMemset2D(d_workspace1, 2 * pitch_nt * sizeof(double), 0.0, 2 * nt * sizeof(double), curr_chunk_size));
        gpuErrchk(cudaMemset(d_workspace2, 0.0, 2 * workspace2_size * sizeof(double)));

        // ***Copy values from host buffer to device workspace2 with pitch_q
        size_t offset = 2 * q_start;                     // host source array offset
        size_t spitch = 2 * (_nx * ny) * sizeof(double); // host source array pitch
        size_t dpitch = 2 * pitch_q * sizeof(double);    // device destination array pitch
        gpuErrchk(cudaMemcpy2D(d_workspace2,
                               dpitch,
                               h_in + offset,
                               spitch,
                               2 * curr_chunk_size * sizeof(double),
                               length,
                               cudaMemcpyHostToDevice));
        
        // ***Transpose complex matrix ({d_workspace2; pitch_q} --> {d_workspace1; pitch_nt})
        transpose_complex_matrix_kernel<<<gridSize_tran1, blockSize_tran>>>((double2 *)d_workspace2,
                                                                            pitch_q,
                                                                            (double2 *)d_workspace1,
                                                                            pitch_nt,
                                                                            curr_chunk_size,
                                                                            length,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM,
                                                                            (length + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        
        // ***Copy values ({d_workspace1; pitch_nt} --> {d_workspace2; pitch_t})
        dpitch = 2 * pitch_t * sizeof(double);
        spitch = 2 * pitch_nt * sizeof(double);
        gpuErrchk(cudaMemcpy2D(d_workspace2,
                               dpitch,
                               d_workspace1,
                               spitch,
                               length * 2 * sizeof(double),
                               curr_chunk_size,
                               cudaMemcpyDeviceToDevice));
        
        // +++ FFT PART +++
        // ***Do fft (d_workspace1 --> d_workspace1)
        cufftSafeCall(cufftExecZ2Z(fft_plan, (CUFFTCOMPLEX *)d_workspace1, (CUFFTCOMPLEX *)d_workspace1, CUFFT_FORWARD));
        
        // ***Compute square modulus (d_workspace1 --> d_workspace1)
        square_modulus_kernel<<<gridSize_sqmod1, blockSize_sqmod>>>((double2 *)d_workspace1,
                                                                    nt,
                                                                    pitch_nt,
                                                                    curr_chunk_size * pitch_nt);
        gpuErrchk(cudaPeekAtLastError());
        
        // ***Do fft (d_workspace1 --> d_workspace1)
        cufftSafeCall(cufftExecZ2Z(fft_plan, (CUFFTCOMPLEX *)d_workspace1, (CUFFTCOMPLEX *)d_workspace1, CUFFT_FORWARD));
        
        // ***Scale fft part (d_workspace1 --> d_workspace1)
        scale_array_kernel<<<gridSize_scale, blockSize_scale>>>(d_workspace1,
                                                                1.0 / (double)nt,
                                                                d_workspace1,
                                                                curr_chunk_size * 2 * pitch_nt);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        // +++ CUMULATIVE SUM PART +++
        // ***Compute square modulus (d_workspace2 --> d_workspace2)
        square_modulus_kernel<<<gridSize_sqmod2, blockSize_sqmod>>>((double2 *)d_workspace2,
                                                                    length,
                                                                    pitch_t,
                                                                    curr_chunk_size * pitch_t);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        // ***Copy the value into the imaginary part of the opposite (with respect to time) element (d_workspace2 --> d_workspace2)
        real2imagopposite_kernel<<<gridSize_sqmod2, blockSize_sqmod>>>((CUFFTCOMPLEX *)d_workspace2,
                                                                       length,
                                                                       pitch_t,
                                                                       curr_chunk_size * pitch_t);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        // ***Compute (exclusive) cumulative sum (prefix scan)
        scan_wrap(d_workspace2,
                  d_workspace2,
                  2 * length,
                  2 * pitch_t,
                  curr_chunk_size);
        
        // gpuErrchk(cudaMemset(d_workspace1, 0.0, 2 * pitch_nt * sizeof(double) * curr_chunk_size));
        // gpuErrchk(cudaMemset(d_workspace2, 0.0, 2 * workspace2_size * sizeof(double)));
        //  ***Linearly combine the two parts (workspace1 + workspace2 --> workspace2)
        linear_combination_final_kernel<<<gridSize_final, blockSize_final>>>((double2 *)d_workspace2,
                                                                             pitch_t,
                                                                             (double2 *)d_workspace1,
                                                                             pitch_nt,
                                                                             (double2 *)d_workspace2,
                                                                             pitch_t,
                                                                             length,
                                                                             curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        // ***Keep only selected lags ({d_workspace2; pitch_t} --> {d_workspace1; pitch_t})
        copy_selected_lags_kernel<<<gridSize_copy, blockSize_copy>>>((double2 *)d_workspace2,
                                                                     (double2 *)d_workspace1,
                                                                     d_lags,
                                                                     lags.size(),
                                                                     pitch_t,
                                                                     pitch_t,
                                                                     curr_chunk_size);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
        
        // ***Transpose complex matrix ({d_workspace1; pitch_t} --> {d_workspace2; pitch_q})
        transpose_complex_matrix_kernel<<<gridSize_tran2, blockSize_tran>>>((double2 *)d_workspace1,
                                                                            pitch_t,
                                                                            (double2 *)d_workspace2,
                                                                            pitch_q,
                                                                            lags.size(),
                                                                            curr_chunk_size,
                                                                            (lags.size() + TILE_DIM - 1) / TILE_DIM,
                                                                            (curr_chunk_size + TILE_DIM - 1) / TILE_DIM);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // ***Copy from device to host (d_workspace2 --> host)
        // elements are treated as complex doubles
        // to speed up transfer, use pitch_q
        offset = 2 * q_start;                     // host destination array offset
        spitch = 2 * pitch_q * sizeof(double);    // host source array pitch
        dpitch = 2 * (_nx * ny) * sizeof(double); // device destination array pitch
        gpuErrchk(cudaMemcpy2D(h_in + offset,
                               dpitch,
                               d_workspace2,
                               spitch,
                               2 * curr_chunk_size * sizeof(double),
                               lags.size(),
                               cudaMemcpyDeviceToHost));
    }

    // ***Free memory
    cufftSafeCall(cufftDestroy(fft_plan));
    gpuErrchk(cudaFree(d_workspace1));
    gpuErrchk(cudaFree(d_workspace2));
    gpuErrchk(cudaFree(d_lags));
}