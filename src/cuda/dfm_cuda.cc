// Maintainer: enrico-lattuada

/*! \file dfm_cuda.cc
    \brief Definition of C++ handlers for Digital Fourier Microscopy functions on the GPU
*/

// *** headers ***
#include "dfm_cuda.h"

#include "dfm_cuda.cuh"
#include "../cpp/helper_dfm.h"

#include <cuda_runtime.h>

// *** code ***

/*!
    Evaluate the device memory pitch for multiple subarrays of size N
 */
size_t get_device_pitch(size_t N)
{
    size_t pitch;
    cudaGetDevicePitch(N, pitch);

    return pitch;
}

/*! \brief Get the device memory for fft2
    \param nx       number of fft nodes in x direction
    \param ny       number of fft nodes in y direction
    \param batch    number of batch elements
 */
size_t get_device_fft2_mem(size_t nx,
                           size_t ny,
                           size_t batch)
{
    // The following line should be changed to move to workstations with multiple GPUs
    size_t memsize[1]; // We are only considering workstations with 1 GPU
    cudaGetFft2MemSize(nx,
                       ny,
                       batch,
                       memsize);

    return memsize[0];
}

/*! \brief Get the device memory for fft
    \param nt       number of fft nodes in t direction
    \param batch    number of batch elements
    \param pitch    pitch of input array
 */
size_t get_device_fft_mem(size_t nt,
                          size_t batch,
                          size_t pitch)
{
    // The following line should be changed to move to workstations with multiple GPUs
    size_t memsize[1]; // We are only considering workstations with 1 GPU
    cudaGetFftMemSize(nt,
                      batch,
                      pitch,
                      memsize);

    return memsize[0];
}

/*!
    Export dfm_cuda functions to python.
 */
void export_dfm_cuda(py::module &m)
{
    m.def("get_device_pitch", &get_device_pitch);
    m.def("get_device_fft2_mem", &get_device_fft2_mem);
    m.def("get_device_fft_mem", &get_device_fft_mem);
}