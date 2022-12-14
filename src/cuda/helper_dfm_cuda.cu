// Maintainer: enrico-lattuada

/*! \file helper_dfm_cuda.cu
    \brief Definition of helper functions for Digital Fourier Microscopy on the GPU
*/

// *** headers ***
#include "helper_dfm_cuda.cuh"
#include "helper_debug.cuh"

#include <cuda_runtime.h>

// *** code ***
const unsigned int TILE_DIM = 32;  // leave this unchanged!
const unsigned int BLOCK_ROWS = 8; // leave this unchanged!

/*!
    Convert array from float to double on device and prepare for fft2 (u_int8_t specialization)
*/
template <typename T>
__global__ void copy_convert_kernel(T *d_in,
                                    double *d_out,
                                    unsigned int width,
                                    unsigned int Npixels,
                                    unsigned int ipitch,
                                    unsigned int idist,
                                    unsigned int opitch,
                                    unsigned int odist,
                                    unsigned int N)
{
    for (unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x; tid < N; tid += blockDim.x * gridDim.x)
    {
        unsigned int t = tid / Npixels;
        unsigned int y = (tid - t * Npixels) / width;
        unsigned int x = tid - t * Npixels - y * width;

        T val = d_in[t * idist + y * ipitch + x];

        d_out[t * odist + y * opitch + x] = (double)val;
    }
}

template __global__ void copy_convert_kernel<double>(double *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<float>(float *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<int64_t>(int64_t *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<int32_t>(int32_t *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<int16_t>(int16_t *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<u_int64_t>(u_int64_t *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<u_int32_t>(u_int32_t *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<u_int16_t>(u_int16_t *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);
template __global__ void copy_convert_kernel<u_int8_t>(u_int8_t *d_in, double *d_out, unsigned int width, unsigned int Npixels, unsigned int ipitch, unsigned int idist, unsigned int opitch, unsigned int odist, unsigned int N);

/*!
    Compute b = A * a
 */
__global__ void scale_array_kernel(double *a,
                                   double A,
                                   double *b,
                                   unsigned int N)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    {
        b[i] = A * a[i];
    }
}

/*!
    Transpose complex matrix with pitch
 */
__global__ void transpose_complex_matrix_kernel(double2 *matIn,
                                                unsigned int ipitch,
                                                double2 *matOut,
                                                unsigned int opitch,
                                                unsigned int width,
                                                unsigned int height)
{
    __shared__ double2 tile[TILE_DIM][TILE_DIM + 1];

    unsigned int i_x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int i_y = blockIdx.y * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

    // load matrix portion into tile
    // every thread loads 4 elements into tile
    unsigned int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (i_x < width && (i_y + i) < height)
        {
            tile[threadIdx.y + i][threadIdx.x] = matIn[(i_y + i) * ipitch + i_x];
        }
    }
    __syncthreads();

    // transpose block offset
    i_x = blockIdx.y * TILE_DIM + threadIdx.x;
    i_y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (i_x < height && (i_y + i) < width)
        {
            matOut[(i_y + i) * opitch + i_x] = tile[threadIdx.x][threadIdx.y + i];
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
                                                unsigned int length,
                                                unsigned int Nlags,
                                                unsigned int Nq,
                                                unsigned int Ntdt,
                                                unsigned int pitch)
{
    // Get current thread index
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < Nq * Ntdt; i += blockDim.x * gridDim.x)
    {
        // Get t1 index
        unsigned int idx_t1 = i / Nq;
        // Get q
        unsigned int q = i - idx_t1 * Nq;

        // Get t1
        unsigned int t1 = __ldg(d_t1 + idx_t1);
        // Get dt
        unsigned int num = __ldg(d_num + t1);
        unsigned int lag = idx_t1 - num;
        unsigned int dt = __ldg(d_lags + lag);

        // Compute t2
        unsigned int t2 = t1 + dt;

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
    Make full power spectrum (copy symmetric part)
*/
__global__ void make_full_powspec_kernel(double2 *d_in,
                                         unsigned int ipitch,
                                         double *d_out,
                                         unsigned int opitch,
                                         unsigned int nxh,
                                         unsigned int nx,
                                         unsigned int ny,
                                         unsigned int N)
{
    __shared__ double2 tile[TILE_DIM][TILE_DIM + 1];

    unsigned int i_x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int i_y = blockIdx.y * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

    // load matrix portion into tile
    // every thread loads 4 elements into tile
    unsigned int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (i_x < nxh && (i_y + i) < ny * N)
        {
            tile[threadIdx.y + i][threadIdx.x] = d_in[(i_y + i) * ipitch + i_x];
        }
    }
    __syncthreads();

    unsigned int curr_y;
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

/*! \brief Shift power spectrum
    \param d_in     input array
    \param ipitch   pitch of input array
    \param d_out    output array
    \param opitch   pitch of output array
    \param nx       number of fft nodes over x
    \param ny       number of fft nodes over y
    \param N        number of 2d matrices
*/
__global__ void shift_powspec_kernel(double *d_in,
                                     unsigned int ipitch,
                                     double *d_out,
                                     unsigned int opitch,
                                     unsigned int nx,
                                     unsigned int ny,
                                     unsigned int N)
{
    __shared__ double tile[TILE_DIM][TILE_DIM + 1];

    unsigned int i_x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int i_y = blockIdx.y * TILE_DIM + threadIdx.y; // threadIdx.y goes from 0 to 7

    // load matrix portion into tile
    // every thread loads 4 elements into tile
    unsigned int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        if (i_x < nx && (i_y + i) < ny * N)
        {
            tile[threadIdx.y + i][threadIdx.x] = d_in[(i_y + i) * ipitch + i_x];
        }
    }
    __syncthreads();

    unsigned int curr_y;
    unsigned int shift_x, shift_y;
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