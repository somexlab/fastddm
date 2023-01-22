// Maintainer: enrico-lattuada

/*! \file helper_ddm_cuda.cu
    \brief Definition of helper functions for Differential Dynamic Microscopy on the GPU
*/

// *** headers ***
#include "helper_ddm_cuda.cuh"
#include "helper_debug.cuh"

#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// *** code ***
const unsigned long long TILE_DIM = 32;  // leave this unchanged!
const unsigned long long BLOCK_ROWS = 8; // leave this unchanged!

//! double2 addition
HOSTDEVICE inline double2& operator+=(double2 &a, const double2 &b)
{
    a.x += b.x;
    a.y += b.y;
    return a;
}

//! double2 division
HOSTDEVICE inline double2 operator/(const double2 &a, const double2 &b)
{
    return make_double2(a.x / b.x, a.y / b.y);
}

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

template __global__ void copy_convert_kernel<u_int8_t>(u_int8_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<int16_t>(int16_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<u_int16_t>(u_int16_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<int32_t>(int32_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<u_int32_t>(u_int32_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<int64_t>(int64_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<u_int64_t>(u_int64_t *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<float>(float *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);
template __global__ void copy_convert_kernel<double>(double *d_in, double *d_out, unsigned long long width, unsigned long long height, unsigned long long length, unsigned long long ipitch, unsigned long long idist, unsigned long long opitch, unsigned long long odist);

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
    Make full power spectrum (copy symmetric part)
*/
__global__ void make_full_powspec_kernel(double2 *d_in,
                                         unsigned long long ipitch,
                                         double *d_out,
                                         unsigned long long opitch,
                                         unsigned long long nxh,
                                         unsigned long long nx,
                                         unsigned long long ny,
                                         unsigned long long N,
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
                if (i_x < nxh && (i_y + i) < ny * N)
                {
                    tile[threadIdx.y + i][threadIdx.x] = d_in[(i_y + i) * ipitch + i_x];
                }
            }
            __syncthreads();

            unsigned long long curr_y;
            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < nxh && (i_y + i) < ny * N)
                {
                    // copy real part (left side)
                    d_out[(i_y + i) * opitch + i_x] = tile[threadIdx.y + i][threadIdx.x].x;

                    // make symmetric part (right side)
                    if (i_x > 0)
                    {
                        curr_y = (i_y + i) % ny;
                        if (curr_y > 0)
                        {
                            d_out[(i_y + i + ny - 2 * curr_y) * opitch + nx - i_x] = tile[threadIdx.y + i][threadIdx.x].x;
                        }
                        else
                        {
                            d_out[(i_y + i) * opitch + nx - i_x] = tile[threadIdx.y + i][threadIdx.x].x;
                        }
                    }
                }
            }
        }
    }
}

/*!
    Shift power spectrum
*/
__global__ void shift_powspec_kernel(double *d_in,
                                     unsigned long long ipitch,
                                     double *d_out,
                                     unsigned long long opitch,
                                     unsigned long long nx,
                                     unsigned long long ny,
                                     unsigned long long N,
                                     unsigned long long NblocksX,
                                     unsigned long long NblocksY)
{
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];

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
            unsigned long long shift_x, shift_y;
            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < nx && (i_y + i) < ny * N)
                {
                    curr_y = (i_y + i) % ny;
                    shift_x = (i_x + nx / 2) % nx;
                    shift_y = (curr_y + ny / 2) % ny;
                    d_out[(i_y + i + shift_y - curr_y) * opitch + shift_x] = tile[threadIdx.y + i][threadIdx.x];
                }
            }
        }
    }
}

/*!
    Compute square modulus of complex array
*/
__global__ void square_modulus_kernel_old(double2 *d_in,
                                          unsigned long long length,
                                          unsigned long long pitch,
                                          unsigned long long N)
{
    for (unsigned long long i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        if (i - (i / pitch) * pitch < length)
        {
            d_in[i].x = d_in[i].x * d_in[i].x + d_in[i].y * d_in[i].y;
            d_in[i].y = 0.0;
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
        double2 tmp_sum = 0.0;
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
    extern __shared__ double2 sdata[];
    unsigned long long blockSize = 2 * blockDim.x;
    unsigned long long tid = threadIdx.x;

    for (unsigned long long q = blockIdx.x; q < Nq; q += gridDim.x)
    {
        double2 tmp_sum = make_double2(0.0, 0.0);
        for (unsigned long long t = tid; t < length; t += blockSize)
        {
            double2 a = d_in[q * pitch + t];
            tmp_sum.x += a.x;
            tmp_sum.y += a.y;

            // Ensure we don't read out of bounds
            if (t + blockDim.x < length)
            {
                a = d_in[q * pitch + t + blockDim.x];
                tmp_sum.x += a.x;
                tmp_sum.y += a.y;
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
            d_out[q] = tmp_sum / make_double2(length, length);
        }
    }
}