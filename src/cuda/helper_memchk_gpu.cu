// Maintainer: enrico-lattuada

/*! \file helper_memchk_gpu.cu
    \brief Definition of CUDA helper functions for memory check and optimization for GPU routines
*/

// *** headers ***
#include "helper_memchk_gpu.cuh"
#include "helper_cufft.cuh"
#include "helper_debug.cuh"

#include <cuda_runtime.h>

// *** code ***

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