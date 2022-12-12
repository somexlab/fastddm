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
