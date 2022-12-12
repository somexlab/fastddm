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
size_t get_device_pitch(size_t N,
                        int Nbytes)
{
    size_t pitch;
    switch (Nbytes)
    {
    case 16:
        cudaGetDevicePitch16B(N, pitch);
        break;
    case 8:
        cudaGetDevicePitch8B(N, pitch);
        break;
    case 4:
        cudaGetDevicePitch4B(N, pitch);
        break;
    case 2:
        cudaGetDevicePitch2B(N, pitch);
        break;
    case 1:
        cudaGetDevicePitch1B(N, pitch);
        break;
    default:
        cudaGetDevicePitch8B(N, pitch);
    }

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
    Compute the image structure function in direct mode
    using differences of fourier transformed images on the GPU.
 */
template <typename T>
py::array_t<double> dfm_direct_cuda(py::array_t<T, py::array::c_style> img_seq,
                                    vector<unsigned int> lags,
                                    size_t nx,
                                    size_t ny,
                                    size_t num_fft2,
                                    size_t buff_pitch,
                                    size_t num_chunks,
                                    size_t pitch_q,
                                    size_t pitch_t)
{
    // ***Get input array and dimensions
    size_t length = img_seq.shape()[0]; // get length of original input
    size_t height = img_seq.shape()[1]; // get height of original input
    size_t width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();    // get input data

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      doubles [the input needs to be twice as large]
     */
    size_t _nx = nx / 2 + 1;
    py::array_t<double> out = py::array_t<double>(2 * _nx * ny * length);
    auto p_out = out.mutable_data();

    // ***Transfer data to GPU and compute fft2
    compute_fft2(p_img_seq,
                 p_out,
                 width,
                 height,
                 length,
                 nx,
                 ny,
                 num_fft2,
                 buff_pitch);

    // ***Compute correlations

    // ***Convert raw output to full and shifted ISF

    // ***Resize output
    // the full size of the image structure function is
    // nx * ny * #(lags)
    out.resize({lags.size(), ny, nx});

    // ***Return result to python
    return out;
}

/*!
    Export dfm_cuda functions to python.
 */
void export_dfm_cuda(py::module &m)
{
    m.def("get_device_pitch", &get_device_pitch);
    m.def("get_device_fft2_mem", &get_device_fft2_mem);
    m.def("get_device_fft_mem", &get_device_fft_mem);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<double>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<float>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<int64_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<int32_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<int16_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint64_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint32_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint16_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint8_t>);
}