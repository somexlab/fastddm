// Maintainer: enrico-lattuada

/*! \file dfm_cuda.cu
    \brief Definition of core CUDA Digital Fourier Microscopy functions
*/

// *** headers ***
#include "dfm_cuda.cuh"

#include "helper_debug.cuh"
#include "helper_cufft.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdlib.h>

// #include <chrono>
// using namespace std::chrono;

#define CUFFTCOMPLEX cufftDoubleComplex

// *** code ***

/*!
    Evaluate the device memory pitch for multiple subarrays of size N
*/
void cudaGetDevicePitch(size_t N,
                        size_t &pitch)
{
    double *d_arr;

    gpuErrchk(cudaMallocPitch(&d_arr, &pitch, N * sizeof(double), 2));

    pitch /= sizeof(double);

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