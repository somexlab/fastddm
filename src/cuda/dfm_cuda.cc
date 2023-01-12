// Maintainer: enrico-lattuada

/*! \file dfm_cuda.cc
    \brief Definition of C++ handlers for Digital Fourier Microscopy functions on the GPU
*/

// *** headers ***
#include "dfm_cuda.h"
#include "dfm_cuda.cuh"

#include "helper_memchk_gpu.h"

#include <cuda_runtime.h>

// *** code ***

/*!
    Compute the image structure function in direct mode
    using differences of fourier transformed images on the GPU.
 */
template <typename T>
py::array_t<double> dfm_direct_cuda(py::array_t<T, py::array::c_style> img_seq,
                                    vector<unsigned int> lags,
                                    unsigned long long nx,
                                    unsigned long long ny)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();    // get input data

    // Check host memory
    chk_host_mem_direct(nx, ny, length, lags);

    // Check device memory and optimize
    unsigned long long num_fft2, num_chunks, num_fullshift;
    unsigned long long pitch_buff, pitch_q, pitch_t, pitch_fs;
    chk_device_mem_direct(width,
                          height,
                          sizeof(T),
                          nx,
                          ny,
                          length,
                          lags,
                          std::is_same<T, double>::value,
                          num_fft2,
                          num_chunks,
                          num_fullshift,
                          pitch_buff,
                          pitch_q,
                          pitch_t,
                          pitch_fs);

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      doubles [the input needs to be twice as large]
     */
    unsigned long long _nx = nx / 2 + 1;
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
                 pitch_buff);

    // ***Compute correlations
    correlate_direct(p_out,
                     lags,
                     length,
                     nx,
                     ny,
                     num_chunks,
                     pitch_q,
                     pitch_t);

    // ***Convert raw output to full and shifted ISF
    // Convert raw output to full and shifted ISF
    make_full_shift(p_out,
                    lags,
                    nx,
                    ny,
                    num_fullshift,
                    pitch_fs);

    // ***Resize output
    // the full size of the image structure function is
    // nx * ny * #(lags)
    out.resize({(unsigned long long)(lags.size()), ny, nx});

    // ***Return result to python
    return out;
}

/*!
    Compute the image structure function in fft mode
    using the Wiener-Khinchin theorem on the GPU.

    Notice that nt must be at least 2*length to avoid
    circular correlation.
 */
template <typename T>
py::array_t<double> dfm_fft_cuda(py::array_t<T, py::array::c_style> img_seq,
                                 vector<unsigned int> lags,
                                 unsigned long long nx,
                                 unsigned long long ny,
                                 unsigned long long nt)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();    // get input data

    // Check host memory
    chk_host_mem_fft(nx, ny, length);

    // Check device memory and optimize
    unsigned long long num_fft2, num_chunks, num_fullshift;
    unsigned long long pitch_buff, pitch_q, pitch_t, pitch_nt, pitch_fs;
    chk_device_mem_fft(width,
                       height,
                       sizeof(T),
                       nx,
                       ny,
                       nt,
                       length,
                       lags,
                       std::is_same<T, double>::value,
                       num_fft2,
                       num_chunks,
                       num_fullshift,
                       pitch_buff,
                       pitch_q,
                       pitch_t,
                       pitch_nt,
                       pitch_fs);

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      doubles [the input needs to be twice as large]
     */
    unsigned long long _nx = nx / 2 + 1;
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
                 pitch_buff);

    // ***Compute correlations
    correlate_fft(p_out,
                  lags,
                  length,
                  nx,
                  ny,
                  nt,
                  num_chunks,
                  pitch_q,
                  pitch_t,
                  pitch_nt);

    // ***Convert raw output to full and shifted ISF
    make_full_shift(p_out,
                    lags,
                    nx,
                    ny,
                    num_fullshift,
                    pitch_fs);

    // ***Resize output
    // the full size of the image structure function is
    // nx * ny * #(lags)
    out.resize({(unsigned long long)(lags.size()), ny, nx});

    // ***Return result to python
    return out;
}

/*!
    Set CUDA device to be used.
    Throws a runtime_error if the device id is out of bounds.
*/
void set_device(int gpu_id)
{
    int dev_num;
    cudaGetDeviceCount(&dev_num);
    if (gpu_id < dev_num)
    {
        int valid_devices[] = {gpu_id};
        cudaSetValidDevices(valid_devices, 1);
    }
    else
    {
        throw std::runtime_error("Device id out of bounds. Choose id < " + to_string(dev_num));
    }
}

/*!
    Export dfm_cuda functions to python.
 */
void export_dfm_cuda(py::module &m)
{
    m.def("set_device", &set_device);
    // m.def("get_device_pitch", &get_device_pitch);
    // m.def("get_device_fft2_mem", &get_device_fft2_mem);
    // m.def("get_device_fft_mem", &get_device_fft_mem);
    // Leave functino export in this order!
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint8_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<int16_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint16_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<int32_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint32_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<int64_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<uint64_t>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<float>);
    m.def("dfm_direct_cuda", &dfm_direct_cuda<double>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<uint8_t>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<int16_t>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<uint16_t>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<int32_t>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<uint32_t>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<int64_t>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<uint64_t>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<float>);
    m.def("dfm_fft_cuda", &dfm_fft_cuda<double>);
}