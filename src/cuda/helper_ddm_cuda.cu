// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file helper_ddm_cuda.cu
    \brief Definition of helper functions for Differential Dynamic Microscopy on the GPU
*/

// *** headers ***
#include "helper_ddm_cuda.cuh"
#include "helper_debug.cuh"

#include <cstdint>

#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// *** code ***
const unsigned long long TILE_DIM = 32;  // leave this unchanged!
const unsigned long long BLOCK_ROWS = 8; // leave this unchanged!

/*!
    Convert array from T to double on device and prepare for fft2
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
                                    unsigned long long odist)
{
    for (unsigned long long t = blockIdx.x; t < length; t += gridDim.x)
    {
        for (unsigned long long y = blockIdx.y; y < height; y += gridDim.y)
        {
            for (unsigned long long x = threadIdx.x; x < width; x += blockDim.x)
            {
                d_out[t * odist + y * opitch + x] = (double)d_in[t * idist + y * ipitch + x];
            }
        }
    }
}

template __global__ void copy_convert_kernel<uint8_t>(uint8_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<int16_t>(int16_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<uint16_t>(uint16_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<int32_t>(int32_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<uint32_t>(uint32_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<int64_t>(int64_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<uint64_t>(uint64_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<float>(float *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<double>(double *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);

/*!
    Apply window function to image sequence
 */
__global__ void apply_window_kernel(double *d_in,
                                    Scalar *window,
                                    unsigned long long width,
                                    unsigned long long height,
                                    unsigned long long length,
                                    unsigned long long ipitch,
                                    unsigned long long idist,
                                    unsigned long long wpitch)
{
    for (unsigned long long t = blockIdx.x; t < length; t += gridDim.x)
    {
        for (unsigned long long y = blockIdx.y; y < height; y += gridDim.y)
        {
            for (unsigned long long x = threadIdx.x; x < width; x += blockDim.x)
            {
                d_in[t * idist + y * ipitch + x] *= (double)window[y * wpitch + x];
            }
        }
    }
}

/*!
    Compute a *= A
 */
__global__ void scale_array_kernel(double2 *a,
                                   unsigned long long pitch,
                                   unsigned long long length,
                                   double A,
                                   unsigned long long N)
{
    for (unsigned long long row = blockIdx.x; row < N; row += gridDim.x)
    {
        for (unsigned long long tid = threadIdx.x; tid < length; tid += blockDim.x)
        {
            a[row * pitch + tid].x *= A;
            a[row * pitch + tid].y *= A;
        }
    }
}

/*!
    Convert 2D array in place from double2 to float2
 */
__global__ void double2float_inplace_kernel(double2 *a,
                                            float2 *b,
                                            unsigned long long pitch,
                                            unsigned long long length,
                                            unsigned long long N)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ float2 tmp2[];

    for (unsigned long long row = blockIdx.x; row < N; row += gridDim.x)
    {
        // copy values to shared memory
        unsigned long long tid = threadIdx.x;
        while (tid < length)
        {
            tmp2[tid] = make_float2(a[row * pitch + tid].x, a[row * pitch + tid].y);
            tid += blockDim.x;
        }

        // synchronize threads
        cg::sync(cta);

        // copy values back to array
        tid = threadIdx.x;
        while (tid < length)
        {
            b[row * pitch * 2 + tid] = tmp2[tid];
            tid += blockDim.x;
        }
    }
}

/*!
    Convert 2D array from double to float
*/
__global__ void double2float_kernel(double *a,
                                    unsigned long long ipitch,
                                    float *b,
                                    unsigned long long opitch,
                                    unsigned long long length,
                                    unsigned long long N)
{
    for (unsigned long long row = blockIdx.x; row < N; row += gridDim.x)
    {
        for (unsigned long long tid = threadIdx.x; tid < length; tid += blockDim.x)
        {
            b[row * opitch + tid] = (float)a[row * ipitch + tid];
        }
    }
}

/*!
    Convert 2D array from float to double
*/
__global__ void float2double_kernel(float *a,
                                    unsigned long long ipitch,
                                    double *b,
                                    unsigned long long opitch,
                                    unsigned long long length,
                                    unsigned long long N)
{
    for (unsigned long long row = blockIdx.x; row < N; row += gridDim.x)
    {
        for (unsigned long long tid = threadIdx.x; tid < length; tid += blockDim.x)
        {
            b[row * opitch + tid] = (double)a[row * ipitch + tid];
        }
    }
}

/*!
    Transpose complex matrix with pitch
 */
__global__ void transpose_complex_matrix_kernel(double2 *matIn,
                                                unsigned long long ipitch,
                                                double2 *matOut,
                                                unsigned long long opitch,
                                                unsigned long long width,
                                                unsigned long long height,
                                                unsigned long long NblocksX,
                                                unsigned long long NblocksY)
{
    __shared__ double2 tile[TILE_DIM][TILE_DIM + 1];

    for (unsigned long long blockIDX = blockIdx.x; blockIDX < NblocksX; blockIDX += gridDim.x)
    {
        for (unsigned long long blockIDY = blockIdx.y; blockIDY < NblocksY; blockIDY += gridDim.y)
        {
            unsigned long long i_x = blockIDX * TILE_DIM + threadIdx.x;
            unsigned long long i_y = blockIDY * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

            // load matrix portion into tile
            // every thread loads 4 elements into tile
            unsigned long long i;
            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < width && (i_y + i) < height)
                {
                    tile[threadIdx.y + i][threadIdx.x] = matIn[(i_y + i) * ipitch + i_x];
                }
            }
            __syncthreads();

            // transpose block offset
            i_x = blockIDY * TILE_DIM + threadIdx.x;
            i_y = blockIDX * TILE_DIM + threadIdx.y;

            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < height && (i_y + i) < width)
                {
                    matOut[(i_y + i) * opitch + i_x] = tile[threadIdx.x][threadIdx.y + i];
                }
            }
        }
    }
}

/*!
    Compute structure function using differences.
    This is inspired by reduce6 from cuda samples.
 */
__global__ void structure_function_diff_kernel(double2 *d_in,
                                               double2 *d_out,
                                               unsigned int *d_lags,
                                               unsigned long long length,
                                               unsigned long long Nlags,
                                               unsigned long long Nq,
                                               unsigned long long pitch)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double sdata[];
    unsigned long long blockSize = 2 * blockDim.x;
    unsigned long long tid = threadIdx.x;

    for (unsigned long long q = blockIdx.y; q < Nq; q += gridDim.y)
    {
        for (unsigned long long dt_idx = blockIdx.x; dt_idx < Nlags; dt_idx += gridDim.x)
        {
            unsigned long long dt = d_lags[dt_idx];

            // Perform first level of reduction
            // Reading from global memory, writing to shared memory
            double tmp_sum = 0.0;
            unsigned long long t = tid;
            while (t < length - dt)
            {
                double2 a = d_in[q * pitch + t];
                double2 b = d_in[q * pitch + t + dt];
                tmp_sum += (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);

                // ensure we don't read out of bounds
                if (t + blockDim.x < length - dt)
                {
                    double2 a = d_in[q * pitch + t + blockDim.x];
                    double2 b = d_in[q * pitch + t + blockDim.x + dt];
                    tmp_sum += (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
                }

                t += blockSize;
            }

            // Each thread puts its local sum into shared memory
            sdata[tid] = tmp_sum;
            cg::sync(cta);

            // do reduction in shared mem
            if ((blockDim.x >= 512) && (tid < 256))
            {
                sdata[tid] = tmp_sum = tmp_sum + sdata[tid + 256];
            }

            cg::sync(cta);

            if ((blockDim.x >= 256) && (tid < 128))
            {
                sdata[tid] = tmp_sum = tmp_sum + sdata[tid + 128];
            }

            cg::sync(cta);

            if ((blockDim.x >= 128) && (tid < 64))
            {
                sdata[tid] = tmp_sum = tmp_sum + sdata[tid + 64];
            }

            cg::sync(cta);

            cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

            if (cta.thread_rank() < 32)
            {
                // Fetch final intermediate sum from 2nd warp
                if (blockDim.x >= 64)
                    tmp_sum += sdata[tid + 32];
                // Reduce final warp using shuffle
                for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
                {
                    tmp_sum += tile32.shfl_down(tmp_sum, offset);
                }
            }

            // Write result for this block to global mem
            // Also normalize by number of occurrences
            // WARNING!!! IF WE ADD EXCLUDED FRAMES, WE NEED TO CHANGE THIS!!
            if (cta.thread_rank() == 0)
            {
                d_out[q * pitch + dt_idx].x = tmp_sum / double(length - dt);
            }
        }
    }
}

/*!
    Average power spectrum of input images
*/
__global__ void average_power_spectrum_kernel(double2 *d_in,
                                              double2 *d_out,
                                              unsigned long long length,
                                              unsigned long long pitch,
                                              unsigned long long Nq)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double sdata[];
    unsigned long long blockSize = 2 * blockDim.x;
    unsigned long long tid = threadIdx.x;

    for (unsigned long long q = blockIdx.x; q < Nq; q += gridDim.x)
    {
        double tmp_sum = 0.0;
        for (unsigned long long t = tid; t < length; t += blockSize)
        {
            double2 a = d_in[q * pitch + t];
            tmp_sum += a.x * a.x + a.y * a.y;

            // ensure we don't read out of bounds
            if (t + blockDim.x < length)
            {
                a = d_in[q * pitch + t + blockDim.x];
                tmp_sum += a.x * a.x + a.y * a.y;
            }
        }

        // Each thread puts its local sum into shared memory
        sdata[tid] = tmp_sum;
        cg::sync(cta);

        // do reduction in shared mem
        if ((blockDim.x >= 512) && (tid < 256))
        {
            sdata[tid] = tmp_sum = tmp_sum + sdata[tid + 256];
        }

        cg::sync(cta);

        if ((blockDim.x >= 256) && (tid < 128))
        {
            sdata[tid] = tmp_sum = tmp_sum + sdata[tid + 128];
        }

        cg::sync(cta);

        if ((blockDim.x >= 128) && (tid < 64))
        {
            sdata[tid] = tmp_sum = tmp_sum + sdata[tid + 64];
        }

        cg::sync(cta);

        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32)
        {
            // Fetch final intermediate sum from 2nd warp
            if (blockDim.x >= 64)
                tmp_sum += sdata[tid + 32];
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
            {
                tmp_sum += tile32.shfl_down(tmp_sum, offset);
            }
        }

        // Write result for this block to global mem
        // Also normalize by number of frames
        // WARNING!!! IF WE ADD EXCLUDED FRAMES, WE NEED TO CHANGE THIS!!
        if (cta.thread_rank() == 0)
        {
            d_out[q].x = tmp_sum / (double)length;
        }
    }
}

/*!
    Average over time of Fourier transformed input images
*/
__global__ void average_complex_kernel(double2 *d_in,
                                       double2 *d_out,
                                       unsigned long long length,
                                       unsigned long long pitch,
                                       unsigned long long Nq)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double2 sdata2[];
    unsigned long long blockSize = 2 * blockDim.x;
    unsigned long long tid = threadIdx.x;

    for (unsigned long long q = blockIdx.x; q < Nq; q += gridDim.x)
    {
        double tmp_sum_r = 0.0;
        double tmp_sum_i = 0.0;
        for (unsigned long long t = tid; t < length; t += blockSize)
        {
            double2 a = d_in[q * pitch + t];
            tmp_sum_r += a.x;
            tmp_sum_i += a.y;

            // Ensure we don't read out of bounds
            if (t + blockDim.x < length)
            {
                a = d_in[q * pitch + t + blockDim.x];
                tmp_sum_r += a.x;
                tmp_sum_i += a.y;
            }
        }

        // Each thread puts its local sum into shared memory
        sdata2[tid].x = tmp_sum_r;
        sdata2[tid].y = tmp_sum_i;
        cg::sync(cta);

        // do reduction in shared mem
        if ((blockDim.x >= 512) && (tid < 256))
        {
            sdata2[tid].x = tmp_sum_r = tmp_sum_r + sdata2[tid + 256].x;
            sdata2[tid].y = tmp_sum_i = tmp_sum_i + sdata2[tid + 256].y;
        }

        cg::sync(cta);

        if ((blockDim.x >= 256) && (tid < 128))
        {
            sdata2[tid].x = tmp_sum_r = tmp_sum_r + sdata2[tid + 128].x;
            sdata2[tid].y = tmp_sum_i = tmp_sum_i + sdata2[tid + 128].y;
        }

        cg::sync(cta);

        if ((blockDim.x >= 128) && (tid < 64))
        {
            sdata2[tid].x = tmp_sum_r = tmp_sum_r + sdata2[tid + 64].x;
            sdata2[tid].y = tmp_sum_i = tmp_sum_i + sdata2[tid + 64].y;
        }

        cg::sync(cta);

        cg::thread_block_tile<32> tile32 = cg::tiled_partition<32>(cta);

        if (cta.thread_rank() < 32)
        {
            // Fetch final intermediate sum from 2nd warp
            if (blockDim.x >= 64)
            {
                tmp_sum_r += sdata2[tid + 32].x;
                tmp_sum_i += sdata2[tid + 32].y;
            }
            // Reduce final warp using shuffle
            for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
            {
                tmp_sum_r += tile32.shfl_down(tmp_sum_r, offset);
                tmp_sum_i += tile32.shfl_down(tmp_sum_i, offset);
            }
        }

        // Write result for this block to global mem
        // Also normalize by number of frames
        // WARNING!!! IF WE ADD EXCLUDED FRAMES, WE NEED TO CHANGE THIS!!
        if (cta.thread_rank() == 0)
        {
            d_out[q].x = tmp_sum_r / (double)length;
            d_out[q].y = tmp_sum_i / (double)length;
        }
    }
}

/*!
    Compute square modulus of complex array
*/
__global__ void square_modulus_kernel(double2 *d_in,
                                      unsigned long long length,
                                      unsigned long long pitch,
                                      unsigned long long N)
{
    for (unsigned long long row = blockIdx.x; row < N; row += gridDim.x)
    {
        for (unsigned long long tid = threadIdx.x; tid < length; tid += blockDim.x)
        {
            unsigned long long i = row * pitch + tid;
            double2 a = d_in[i];
            d_in[i].x = a.x * a.x + a.y * a.y;
            d_in[i].y = 0.0;
        }
    }
}

/*!
    Linear combination c = A * a + B * b
*/
__global__ void linear_combination_kernel(double2 *c,
                                          double2 *a,
                                          double2 A,
                                          double2 *b,
                                          double2 B,
                                          unsigned int N)
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;

    if (tid < N)
    {
        c[tid].x = A.x * a[tid].x + B.x * b[tid].x;
        c[tid].y = A.y * a[tid].y + B.y * b[tid].y;
    }
}

/*!
    Shift power spectrum
*/
__global__ void shift_powspec_kernel(Scalar2 *d_in,
                                     unsigned long long ipitch,
                                     Scalar *d_out,
                                     unsigned long long opitch,
                                     unsigned long long nx,
                                     unsigned long long ny,
                                     unsigned long long N,
                                     unsigned long long NblocksX,
                                     unsigned long long NblocksY)
{
    __shared__ Scalar2 tile[TILE_DIM][TILE_DIM + 1];

    for (unsigned long long blockIDX = blockIdx.x; blockIDX < NblocksX; blockIDX += gridDim.x)
    {
        for (unsigned long long blockIDY = blockIdx.y; blockIDY < NblocksY; blockIDY += gridDim.y)
        {
            unsigned long long i_x = blockIDX * TILE_DIM + threadIdx.x;
            unsigned long long i_y = blockIDY * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

            // load matrix portion into tile
            // every thread loads 4 elements into tile
            unsigned long long i;
            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < nx && (i_y + i) < ny * N)
                {
                    tile[threadIdx.y + i][threadIdx.x] = d_in[(i_y + i) * ipitch + i_x];
                }
            }
            __syncthreads();

            unsigned long long curr_y;
            unsigned long long shift_y;
            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < nx && (i_y + i) < ny * N)
                {
                    curr_y = (i_y + i) % ny;
                    shift_y = (curr_y + ny / 2) % ny;
                    d_out[(i_y + i + shift_y - curr_y) * opitch + i_x] = tile[threadIdx.y + i][threadIdx.x].x;
                }
            }
        }
    }
}

/*!
    Do final linear combination c[i] = (a[0] - b[i].x - 2 * a[i]) / (length - i)
 */
__global__ void linear_combination_final_kernel(double2 *c,
                                                unsigned long long pitch_c,
                                                double2 *a,
                                                unsigned long long pitch_a,
                                                double2 *b,
                                                unsigned long long pitch_b,
                                                unsigned long long length,
                                                unsigned long long N)
{
    __shared__ double a0;

    for (unsigned long long blockID = blockIdx.x; blockID < N; blockID += gridDim.x)
    {
        unsigned long long blockOffset_a = blockID * pitch_a;
        unsigned long long blockOffset_b = blockID * pitch_b;
        unsigned long long blockOffset_c = blockID * pitch_c;

        // read first value of fft corr
        a0 = 2.0 * a[blockOffset_a].x;
        __syncthreads();

        for (unsigned long long threadID = threadIdx.x; threadID < length; threadID += blockDim.x)
        {
            double da = b[blockOffset_b + threadID].x;
            double dc = a[blockOffset_a + threadID].x;
            c[blockOffset_c + threadID].x = (a0 - da - 2 * dc) / (double)(length - threadID);
        }
    }
}

/*!
    Keep only selected lags
*/
__global__ void copy_selected_lags_kernel(double2 *d_in,
                                          double2 *d_out,
                                          unsigned int *d_lags,
                                          unsigned long long Nlags,
                                          unsigned long long ipitch,
                                          unsigned long long opitch,
                                          unsigned long long N)
{
    for (unsigned long long blockID = blockIdx.x; blockID < N; blockID += gridDim.x)
    {
        unsigned long long blockOffsetIn = blockID * ipitch;
        unsigned long long blockOffsetOut = blockID * opitch;

        for (unsigned long long threadID = threadIdx.x; threadID < Nlags; threadID += blockDim.x)
        {
            d_out[blockOffsetOut + threadID] = d_in[blockOffsetIn + d_lags[threadID]];
        }
    }
}

/*!
    Copy real part of element into imaginary part of opposite element
 */
__global__ void real2imagopposite_kernel(double2 *d_arr,
                                         unsigned long long length,
                                         unsigned long long pitch,
                                         unsigned long long N)
{
    for (unsigned long long row = blockIdx.x; row < N; row += gridDim.x)
    {
        for (unsigned long long tid = threadIdx.x; tid < length; tid += blockDim.x)
        {
            unsigned long long opp_idx = length - tid - 1;
            d_arr[row * pitch + tid].y = d_arr[row * pitch + opp_idx].x;
        }
    }
}
