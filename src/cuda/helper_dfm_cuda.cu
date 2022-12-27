// Maintainer: enrico-lattuada

/*! \file helper_dfm_cuda.cu
    \brief Definition of helper functions for Digital Fourier Microscopy on the GPU
*/

// *** headers ***
#include "helper_dfm_cuda.cuh"
#include "helper_debug.cuh"

#include <cuda_runtime.h>

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

// *** code ***
const unsigned long int TILE_DIM = 32;  // leave this unchanged!
const unsigned long int BLOCK_ROWS = 8; // leave this unchanged!

/*!
    Convert array from float to double on device and prepare for fft2 (u_int8_t specialization)
*/
template <typename T>
__global__ void copy_convert_kernel(T *d_in,
                                    double *d_out,
                                    unsigned long int width,
                                    unsigned long int Npixels,
                                    unsigned long int ipitch,
                                    unsigned long int idist,
                                    unsigned long int opitch,
                                    unsigned long int odist,
                                    unsigned long int N)
{
    for (unsigned long int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
    {
        unsigned long int t = tid / Npixels;
        unsigned long int y = (tid - t * Npixels) / width;
        unsigned long int x = tid - t * Npixels - y * width;

        T val = d_in[t * idist + y * ipitch + x];

        d_out[t * odist + y * opitch + x] = (double)val;
    }
}

template __global__ void copy_convert_kernel<double>(double *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<float>(float *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<int64_t>(int64_t *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<int32_t>(int32_t *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<int16_t>(int16_t *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<u_int64_t>(u_int64_t *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<u_int32_t>(u_int32_t *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<u_int16_t>(u_int16_t *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);
template __global__ void copy_convert_kernel<u_int8_t>(u_int8_t *d_in, double *d_out, unsigned long int width, unsigned long int Npixels, unsigned long int ipitch, unsigned long int idist, unsigned long int opitch, unsigned long int odist, unsigned long int N);

/*!
    Compute b = A * a
 */
__global__ void scale_array_kernel(double *a,
                                   double A,
                                   double *b,
                                   unsigned long int N)
{
    for (unsigned long int i = (unsigned long int)blockIdx.x * (unsigned long int)blockDim.x + (unsigned long int)threadIdx.x; i < N; i += (unsigned long int)blockDim.x * (unsigned long int)gridDim.x)
    {
        b[i] = A * a[i];
    }
}

/*!
    Transpose complex matrix with pitch
 */
__global__ void transpose_complex_matrix_kernel(double2 *matIn,
                                                unsigned long int ipitch,
                                                double2 *matOut,
                                                unsigned long int opitch,
                                                unsigned long int width,
                                                unsigned long int height,
                                                unsigned long int NblocksX,
                                                unsigned long int NblocksY)
{
    __shared__ double2 tile[TILE_DIM][TILE_DIM + 1];

    for (unsigned long int blockIDX = blockIdx.x; blockIDX < NblocksX; blockIDX += gridDim.x)
    {
        for (unsigned long int blockIDY = blockIdx.y; blockIDY < NblocksY; blockIDY += gridDim.y)
        {
            unsigned long int i_x = blockIDX * TILE_DIM + threadIdx.x;
            unsigned long int i_y = blockIDY * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

            // load matrix portion into tile
            // every thread loads 4 elements into tile
            unsigned long int i;
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
    Compute correlation using differences
 */
__global__ void correlatewithdifferences_kernel(double2 *d_in,
                                                double2 *d_out,
                                                unsigned int *d_lags,
                                                unsigned int *d_t1,
                                                unsigned int *d_num,
                                                unsigned long int length,
                                                unsigned long int Nlags,
                                                unsigned long int Nq,
                                                unsigned long int Ntdt,
                                                unsigned long int pitch)
{
    // Get current thread index
    for (unsigned long int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nq * Ntdt; i += blockDim.x * gridDim.x)
    {
        // Get t1 index
        unsigned long int idx_t1 = i / Nq;
        // Get q
        unsigned long int q = i - idx_t1 * Nq;

        // Get t1
        unsigned long int t1 = (unsigned long int)(__ldg(d_t1 + idx_t1));
        // Get dt
        unsigned long int num = (unsigned long int)(__ldg(d_num + t1));
        unsigned long int lag = idx_t1 - num;
        unsigned long int dt = (unsigned long int)(__ldg(d_lags + lag));

        // Compute t2
        unsigned long int t2 = t1 + dt;

        // Get pixel values at t1 and t2
        double2 a = d_in[q * pitch + t1];
        double2 b = d_in[q * pitch + t2];

        // Compute square modulus of difference
        double smd = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);

        // Also normalize by number of occurrences
        // WARNING!!! IF WE ADD EXCLUDED FRAMES, WE NEED TO REMOVE THIS!!
        smd /= double(length - dt);

        // Add to output vector
        atomicAdd(&(d_out[q * pitch + lag].x), smd);
    }
}

/*!
    Compute correlation using differences.
    This is inspired by reduce6 from cuda samples.
 */
__global__ void correlate_with_differences_kernel(double2 *d_in,
                                                  double2 *d_out,
                                                  unsigned int *d_lags,
                                                  unsigned long int length,
                                                  unsigned long int Nlags,
                                                  unsigned long int Nq,
                                                  unsigned long int pitch)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    extern __shared__ double sdata[];
    unsigned long int blockSize = 2 * blockDim.x;
    unsigned long int tid = threadIdx.x;

    for (unsigned long int q = blockIdx.y; q < Nq; q += gridDim.y)
    {
        for (unsigned long int dt_idx = blockIdx.x; dt_idx < Nlags; dt += gridDim.x)
        {
            unsigned long int dt = d_lags[dt_idx];

            // Perform first level of reduction
            // Reading from global memory, writing to shared memory
            double tmp_sum = 0.0;
            unsigned long int t = tid;
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
                d_out[q * pitch + dt].x = tmp_sum / double(length - dt);
        }
    }
}

/*!
    Make full power spectrum (copy symmetric part)
*/
__global__ void make_full_powspec_kernel(double2 *d_in,
                                         unsigned long int ipitch,
                                         double *d_out,
                                         unsigned long int opitch,
                                         unsigned long int nxh,
                                         unsigned long int nx,
                                         unsigned long int ny,
                                         unsigned long int N,
                                         unsigned long int NblocksX,
                                         unsigned long int NblocksY)
{
    __shared__ double2 tile[TILE_DIM][TILE_DIM + 1];

    for (unsigned long int blockIDX = blockIdx.x; blockIDX < NblocksX; blockIDX += gridDim.x)
    {
        for (unsigned long int blockIDY = blockIdx.y; blockIDY < NblocksY; blockIDY += gridDim.y)
        {
            unsigned long int i_x = blockIDX * TILE_DIM + threadIdx.x;
            unsigned long int i_y = blockIDY * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

            // load matrix portion into tile
            // every thread loads 4 elements into tile
            unsigned long int i;
            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < nxh && (i_y + i) < ny * N)
                {
                    tile[threadIdx.y + i][threadIdx.x] = d_in[(i_y + i) * ipitch + i_x];
                }
            }
            __syncthreads();

            unsigned long int curr_y;
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
                                     unsigned long int ipitch,
                                     double *d_out,
                                     unsigned long int opitch,
                                     unsigned long int nx,
                                     unsigned long int ny,
                                     unsigned long int N,
                                     unsigned long int NblocksX,
                                     unsigned long int NblocksY)
{
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];

    for (unsigned long int blockIDX = blockIdx.x; blockIDX < NblocksX; blockIDX += gridDim.x)
    {
        for (unsigned long int blockIDY = blockIdx.y; blockIDY < NblocksY; blockIDY += gridDim.y)
        {
            unsigned long int i_x = blockIDX * TILE_DIM + threadIdx.x;
            unsigned long int i_y = blockIDY * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

            // load matrix portion into tile
            // every thread loads 4 elements into tile
            unsigned long int i;
            for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
            {
                if (i_x < nx && (i_y + i) < ny * N)
                {
                    tile[threadIdx.y + i][threadIdx.x] = d_in[(i_y + i) * ipitch + i_x];
                }
            }
            __syncthreads();

            unsigned long int curr_y;
            unsigned long int shift_x, shift_y;
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
__global__ void square_modulus_kernel(double2 *d_in,
                                      unsigned long int length,
                                      unsigned long int pitch,
                                      unsigned long int N)
{
    for (unsigned long int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        if (i - (i / pitch) * pitch < length)
        {
            d_in[i].x = d_in[i].x * d_in[i].x + d_in[i].y * d_in[i].y;
            d_in[i].y = 0.0;
        }
    }
}

/*!
    Copy real part of element into imaginary part of opposite element
 */
__global__ void real2imagopposite_kernel(double2 *d_arr,
                                         unsigned long int length,
                                         unsigned long int pitch,
                                         unsigned long int N)
{
    for (unsigned long int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        unsigned long int el = i / pitch;
        if (i - el * pitch < length)
        {
            unsigned long int idx = el * pitch + (length - 1) - (i - el * pitch);
            d_arr[i].y = d_arr[idx].x;
        }
    }
}

/*!
    Do final linear combination c[i] = (a[0] - b[i].x - 2 * a[i]) / (length - i)
 */
__global__ void linear_combination_final_kernel(double2 *c,
                                                unsigned long int pitch_c,
                                                double2 *a,
                                                unsigned long int pitch_a,
                                                double2 *b,
                                                unsigned long int pitch_b,
                                                unsigned long int length,
                                                unsigned long int N)
{
    __shared__ double a0;

    for (unsigned long int blockID = blockIdx.x; blockID < N; blockID += gridDim.x)
    {
        unsigned long int blockOffset_a = blockID * pitch_a;
        unsigned long int blockOffset_b = blockID * pitch_b;
        unsigned long int blockOffset_c = blockID * pitch_c;

        // read first value of fft corr
        a0 = 2.0 * a[blockOffset_a].x;
        __syncthreads();

        for (unsigned long int threadID = threadIdx.x; threadID < length; threadID += blockDim.x)
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
                                          unsigned long int Nlags,
                                          unsigned long int ipitch,
                                          unsigned long int opitch,
                                          unsigned long int N)
{
    for (unsigned long int blockID = blockIdx.x; blockID < N; blockID += gridDim.x)
    {
        unsigned long int blockOffsetIn = blockID * ipitch;
        unsigned long int blockOffsetOut = blockID * opitch;

        for (unsigned long int threadID = threadIdx.x; threadID < Nlags; threadID += blockDim.x)
        {
            d_out[blockOffsetOut + threadID] = d_in[blockOffsetIn + (unsigned long int)(d_lags[threadID])];
        }
    }
}
