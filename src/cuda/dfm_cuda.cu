// Maintainer: enrico-lattuada

/*! \file dfm_cuda.cu
    \brief Definition of core CUDA Digital Fourier Microscopy functions
*/

// *** headers ***
#include "dfm_cuda.cuh"

#include "helper_debug.cuh"
#include "helper_cufft.cuh"
#include "helper_dfm_cuda.cuh"

#include <cuda_runtime.h>
#include <cufft.h>

#include <stdlib.h>

// #include <chrono>
// using namespace std::chrono;

#define CUFFTCOMPLEX cufftDoubleComplex

// *** code ***

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
