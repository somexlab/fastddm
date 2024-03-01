// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file helper_prefix_sum.cu
    \brief Definition of helper functions for cumulative sum (prefix sum) on GPU
*/

// *** headers ***
#include "helper_prefix_sum.cuh"
#include "helper_debug.cuh"

#include <cuda_runtime.h>

// definitions
#define NUM_BANKS 32ULL
#define LOG_NUM_BANKS 5ULL
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)

unsigned long long BLOCK_SIZE = 512;
unsigned long long ELEMENTS_PER_BLOCK = BLOCK_SIZE * 2;

// *** code ***

/*!
    Compute next power of two larger or equal to n
 */
unsigned long long nextPowerOfTwo(unsigned long long n)
{
    unsigned long long power = 1;
    while (power < n)
    {
        power *= 2;
    }
    return power;
}

// The following implementation is based on:
// https://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/scan/doc/scan.pdf
// https://github.com/lxxue/prefix_sum/blob/master/prefix_sum.cu

/*!
    Scan multiple large arrays on the GPU
 */
void scanManyLargeArrays(double *output,
                         double *input,
                         unsigned long long length,
                         unsigned long long dist,
                         unsigned long long N)
{
    unsigned long long Nx = length / ELEMENTS_PER_BLOCK;             // number of even blocks per row
    unsigned long long remainder = length - Nx * ELEMENTS_PER_BLOCK; // remainder from even blocks

    if (remainder == 0)
    {
        scanManyLargeEvenArrays(output,
                                input,
                                length,
                                dist,
                                Nx,
                                N);
    }
    else
    {
        unsigned long long length_even = length - remainder;

        // copy the last element of the even part of the subarray before prefix sum
        double *a1;
        gpuErrchk(cudaMalloc(&a1, N * sizeof(double)));
        int numSMs;
        gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
        int gridSize_copy = min(N, 32ULL * numSMs);
        copy_every_kernel<<<gridSize_copy, 1>>>(a1,
                                                input + length_even - 1,
                                                dist,
                                                N);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // scan over even arrays
        scanManyLargeEvenArrays(output,
                                input,
                                length_even,
                                dist,
                                Nx,
                                N);

        // scan the remaining elements
        scanManySmallArrays(output + length_even,
                            input + length_even,
                            remainder,
                            dist,
                            N);

        // copy the last element of the even part of the subarray after prefix sum
        double *a2;
        gpuErrchk(cudaMalloc(&a2, N * sizeof(double)));
        copy_every_kernel<<<gridSize_copy, 1>>>(a2,
                                                output + length_even - 1,
                                                dist,
                                                N);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // sum the last element of the even arrays portion
        // and the first element of the remainder input
        // to the remainder scan
        add_many_kernel<<<gridSize_copy, remainder>>>(output + length_even,
                                                      dist,
                                                      1,
                                                      N,
                                                      remainder,
                                                      a1,
                                                      a2);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        // free memory
        gpuErrchk(cudaFree(a1));
        gpuErrchk(cudaFree(a2));
    }
}

/*!
    Scan multiple large even arrays on the GPU
 */
void scanManyLargeEvenArrays(double *output,
                             double *input,
                             unsigned long long length,
                             unsigned long long dist,
                             unsigned long long Nx,
                             unsigned long long N)
{
    // compute execution parameters
    const unsigned long long shared_mem_size = ELEMENTS_PER_BLOCK * sizeof(double);
    int numSMs;
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    int gridSize = min(Nx * N, 32ULL * numSMs);

    // allocate partial sums and incr arrays
    double *sums, *incr;
    gpuErrchk(cudaMalloc(&sums, Nx * N * sizeof(double)));
    gpuErrchk(cudaMalloc(&incr, Nx * N * sizeof(double)));

    // do scan
    prescan_many_even_kernel<<<gridSize, BLOCK_SIZE, 2 * shared_mem_size>>>(output,
                                                                            input,
                                                                            dist,
                                                                            Nx,
                                                                            Nx * N,
                                                                            ELEMENTS_PER_BLOCK,
                                                                            sums);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // compute increment
    unsigned long long scan_sum_threads_needed = (Nx + 1) / 2;
    if (scan_sum_threads_needed > BLOCK_SIZE)
    {
        scanManyLargeArrays(incr,
                            sums,
                            Nx,
                            Nx,
                            N);
    }
    else
    {
        scanManySmallArrays(incr,
                            sums,
                            Nx,
                            Nx,
                            N);
    }

    // add increment
    add_many_kernel<<<gridSize, ELEMENTS_PER_BLOCK>>>(output,
                                                      dist,
                                                      Nx,
                                                      Nx * N,
                                                      ELEMENTS_PER_BLOCK,
                                                      incr);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // free memory
    gpuErrchk(cudaFree(sums));
    gpuErrchk(cudaFree(incr));
}

/*!
    Scan multiple small arrays on the GPU
 */
void scanManySmallArrays(double *output,
                         double *input,
                         unsigned long long length,
                         unsigned long long dist,
                         unsigned long long N)
{
    unsigned long long powerOfTwo = nextPowerOfTwo(length);

    // Compute execution parameters
    const unsigned long long shared_mem_size = powerOfTwo * sizeof(double);
    int numSMs;
    gpuErrchk(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0));
    int gridSize = min(N, 32ULL * numSMs);
    int blockSize = (length + 1) / 2;

    // do scan
    prescan_many_arbitrary_kernel<<<gridSize, blockSize, 2 * shared_mem_size>>>(output,
                                                                                input,
                                                                                dist,
                                                                                N,
                                                                                length,
                                                                                powerOfTwo);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

/*!
    Wrapper function to compute cumulative sum on multiple subarrays on the GPU
 */
void scan_wrap(double *output,
               double *input,
               unsigned long long length,
               unsigned long long dist,
               unsigned long long N)
{
    if (length > ELEMENTS_PER_BLOCK)
    {
        scanManyLargeArrays(output,
                            input,
                            length,
                            dist,
                            N);
    }
    else
    {
        scanManySmallArrays(output,
                            input,
                            length,
                            dist,
                            N);
    }
}

/*!
    Step 1 of cumsummany:
    Compute cumulative sum on multiple even portions of the subarrays on the GPU
 */
__global__ void prescan_many_even_kernel(double *output,
                                         double *input,
                                         unsigned long long dist,
                                         unsigned long long Nx,
                                         unsigned long long NxNy,
                                         unsigned long long n,
                                         double *sums)
{
    extern __shared__ double temp[];

    for (unsigned long long blockID = blockIdx.x; blockID < NxNy; blockID += gridDim.x)
    {
        unsigned long long threadID = threadIdx.x;
        unsigned long long y = (blockID / Nx);
        unsigned long long blockOffset = y * dist + (blockID - y * Nx) * n;

        // load input into shared memory
        unsigned long long ai = threadID;
        unsigned long long bi = threadID + (n / 2);
        unsigned long long bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        unsigned long long bankOffsetB = CONFLICT_FREE_OFFSET(bi);
        temp[ai + bankOffsetA] = input[blockOffset + ai];
        temp[bi + bankOffsetB] = input[blockOffset + bi];

        // build sum in place up the tree
        unsigned long long offset = 1;
        for (unsigned long long d = n >> 1; d > 0; d >>= 1)
        {
            __syncthreads();
            if (threadID < d)
            {
                unsigned long long ai = offset * (2 * threadID + 1) - 1;
                unsigned long long bi = offset * (2 * threadID + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                temp[bi] += temp[ai];
            }
            offset *= 2;
        }
        __syncthreads();

        // write total sum to sums and clear the last element
        if (threadID == 0)
        {
            sums[blockID] = temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)];
            temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0.0;
        }

        // traverse down tree and build scan
        for (unsigned long long d = 1; d < n; d *= 2)
        {
            offset >>= 1;
            __syncthreads();
            if (threadID < d)
            {
                unsigned long long ai = offset * (2 * threadID + 1) - 1;
                unsigned long long bi = offset * (2 * threadID + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                double t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();

        output[blockOffset + ai] = temp[ai + bankOffsetA];
        output[blockOffset + bi] = temp[bi + bankOffsetB];
    }
}

/*!
    Step 2 of cumsummany:
    Compute cumulative sum on multiple arbitrary (small) portions of the subarrays on the GPU
 */
__global__ void prescan_many_arbitrary_kernel(double *output,
                                              double *input,
                                              unsigned long long dist,
                                              unsigned long long N,
                                              unsigned long long n,
                                              unsigned long long powerOfTwo)
{
    extern __shared__ double temp[];

    for (unsigned long long blockID = blockIdx.x; blockID < N; blockID += gridDim.x)
    {
        unsigned long long threadID = threadIdx.x;
        unsigned long long blockOffset = blockID * dist;

        // load input into shared memory
        unsigned long long ai = threadID;
        unsigned long long bi = threadID + (n / 2);
        unsigned long long bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        unsigned long long bankOffsetB = CONFLICT_FREE_OFFSET(bi);

        if (threadID < n)
        {
            temp[ai + bankOffsetA] = input[blockOffset + ai];
            temp[bi + bankOffsetB] = input[blockOffset + bi];
        }
        else
        {
            temp[ai + bankOffsetA] = 0.0;
            temp[bi + bankOffsetB] = 0.0;
        }

        // build sum in place up the tree
        unsigned long long offset = 1;
        for (unsigned long long d = powerOfTwo >> 1; d > 0; d >>= 1)
        {
            __syncthreads();
            if (threadID < d)
            {
                unsigned long long ai = offset * (2 * threadID + 1) - 1;
                unsigned long long bi = offset * (2 * threadID + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                temp[bi] += temp[ai];
            }
            offset *= 2;
        }
        __syncthreads();

        // clear the last element
        if (threadID == 0)
        {
            temp[powerOfTwo - 1 + CONFLICT_FREE_OFFSET(powerOfTwo - 1)] = 0;
        }

        // traverse down tree and build scan
        for (unsigned long long d = 1; d < powerOfTwo; d *= 2)
        {
            offset >>= 1;
            __syncthreads();
            if (threadID < d)
            {
                unsigned long long ai = offset * (2 * threadID + 1) - 1;
                unsigned long long bi = offset * (2 * threadID + 2) - 1;
                ai += CONFLICT_FREE_OFFSET(ai);
                bi += CONFLICT_FREE_OFFSET(bi);

                double t = temp[ai];
                temp[ai] = temp[bi];
                temp[bi] += t;
            }
        }
        __syncthreads();

        if (threadID < n)
        {
            output[blockOffset + ai] = temp[ai + bankOffsetA];
            output[blockOffset + bi] = temp[bi + bankOffsetB];
        }
    }
}

__global__ void add_many_kernel(double *output,
                                unsigned long long dist,
                                unsigned long long Nx,
                                unsigned long long NxNy,
                                unsigned long long n,
                                double *a)
{
    for (unsigned long long blockID = blockIdx.x; blockID < NxNy; blockID += gridDim.x)
    {
        unsigned long long threadID = threadIdx.x;
        if (threadID < n)
        {
            unsigned long long y = (blockID / Nx);
            unsigned long long blockOffset = y * dist + (blockID - y * Nx) * n;

            output[blockOffset + threadID] += a[blockID];
        }
    }
}

__global__ void add_many_kernel(double *output,
                                unsigned long long dist,
                                unsigned long long Nx,
                                unsigned long long NxNy,
                                unsigned long long n,
                                double *a1,
                                double *a2)
{
    for (unsigned long long blockID = blockIdx.x; blockID < NxNy; blockID += gridDim.x)
    {
        unsigned long long threadID = threadIdx.x;
        if (threadID < n)
        {
            unsigned long long y = (blockID / Nx);
            unsigned long long blockOffset = y * dist + (blockID - y * Nx) * n;

            output[blockOffset + threadID] += a1[blockID] + a2[blockID];
        }
    }
}

__global__ void copy_every_kernel(double *output,
                                  double *input,
                                  unsigned long long dist,
                                  unsigned long long N)
{
    for (unsigned long long blockID = blockIdx.x; blockID < N; blockID += gridDim.x)
    {
        output[blockID] = input[blockID * dist];
    }
}
