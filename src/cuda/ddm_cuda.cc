/// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file ddm_cuda.cc
    \brief Definition of C++ handlers for Differential Dynamic Microscopy functions on the GPU
*/

// *** headers ***
#include "ddm_cuda.h"
#include "ddm_cuda.cuh"

#include "helper_memchk_gpu.h"

#include <cuda_runtime.h>

// *** code ***

/*!
    Compute the image structure function in diff mode
    using differences of Fourier transformed images on the GPU.
 */
template <typename T>
py::array_t<Scalar> ddm_diff_cuda(py::array_t<T, py::array::c_style> img_seq,
                                  vector<unsigned int> lags,
                                  unsigned long long nx,
                                  unsigned long long ny,
                                  py::array_t<Scalar, py::array::c_style> window)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();                // get input data

    // ***Get window array
    unsigned long long window_length = window.shape()[0]; // get length of window array
    auto p_window = window.data();
    bool is_window = window_length > 0; // true if window is not empty

    // Check host memory
    chk_host_mem_diff(nx, ny, length, lags.size());

    // Check device memory and optimize
    unsigned long long num_fft2, num_chunks, num_shift;
    unsigned long long pitch_buff, pitch_nx, pitch_q, pitch_t, pitch_fs;
    chk_device_mem_diff(width,
                        height,
                        sizeof(T),
                        nx,
                        ny,
                        length,
                        lags,
                        std::is_same<T, Scalar>::value,
                        is_window,
                        num_fft2,
                        num_chunks,
                        num_shift,
                        pitch_buff,
                        pitch_nx,
                        pitch_q,
                        pitch_t,
                        pitch_fs);

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is [ny * (nx // 2 + 1)] complex
      Scalar [the input needs to be twice as large]
     */
    unsigned long long _nx = nx / 2 + 1;
    unsigned long long dim_t = max(length, (unsigned long long)(lags.size() + 2));
    py::array_t<Scalar> out = py::array_t<Scalar>(2 * _nx * ny * dim_t);
    auto p_out = out.mutable_data();

    // ***Transfer data to GPU and compute fft2
    compute_fft2(p_img_seq,
                 p_out,
                 p_window,
                 is_window,
                 width,
                 height,
                 length,
                 nx,
                 ny,
                 num_fft2,
                 pitch_buff,
                 pitch_nx);

    // ***Compute image structure function
    structure_function_diff(p_out,
                            lags,
                            length,
                            nx,
                            ny,
                            num_chunks,
                            pitch_q,
                            pitch_t);

    // ***Convert raw output to shifted image structure function
    make_shift(p_out,
               lags.size() + 2,
               nx,
               ny,
               num_shift,
               pitch_fs);

    // ***Resize output
    // the full size of the image structure function is
    // nx * ny * #(lags)
    out.resize({(unsigned long long)(lags.size() + 2), ny, _nx});

    // release pointer to output array
    p_out = NULL;

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
py::array_t<Scalar> ddm_fft_cuda(py::array_t<T, py::array::c_style> img_seq,
                                 vector<unsigned int> lags,
                                 unsigned long long nx,
                                 unsigned long long ny,
                                 unsigned long long nt,
                                 py::array_t<Scalar, py::array::c_style> window)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();                // get input data

    // ***Get window array
    unsigned long long window_length = window.shape()[0]; // get length of window array
    auto p_window = window.data();
    bool is_window = window_length > 0; // true if window is not empty

    // Check host memory
    chk_host_mem_fft(nx, ny, length, lags.size());

    // Check device memory and optimize
    unsigned long long num_fft2, num_chunks, num_shift;
    unsigned long long pitch_buff, pitch_nx, pitch_q, pitch_t, pitch_nt, pitch_fs;
    chk_device_mem_fft(width,
                       height,
                       sizeof(T),
                       nx,
                       ny,
                       nt,
                       length,
                       lags,
                       std::is_same<T, Scalar>::value,
                       is_window,
                       num_fft2,
                       num_chunks,
                       num_shift,
                       pitch_buff,
                       pitch_nx,
                       pitch_q,
                       pitch_t,
                       pitch_nt,
                       pitch_fs);

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is [ny * (nx // 2 + 1)] complex
      Scalar [the input needs to be twice as large]
     */
    unsigned long long _nx = nx / 2 + 1;
    unsigned long long dim_t = max(length, (unsigned long long)(lags.size() + 2));
    py::array_t<Scalar> out = py::array_t<Scalar>(2 * _nx * ny * dim_t);
    auto p_out = out.mutable_data();

    // ***Transfer data to GPU and compute fft2
    compute_fft2(p_img_seq,
                 p_out,
                 p_window,
                 is_window,
                 width,
                 height,
                 length,
                 nx,
                 ny,
                 num_fft2,
                 pitch_buff,
                 pitch_nx);

    // ***Compute image structure function
    structure_function_fft(p_out,
                           lags,
                           length,
                           nx,
                           ny,
                           nt,
                           num_chunks,
                           pitch_q,
                           pitch_t,
                           pitch_nt);

    // ***Convert raw output to shifted image structure function
    make_shift(p_out,
               lags.size() + 2,
               nx,
               ny,
               num_shift,
               pitch_fs);

    // ***Resize output
    // the full size of the image structure function is
    // nx * ny * #(lags)
    out.resize({(unsigned long long)(lags.size() + 2), ny, _nx});

    // release pointer to output array
    p_out = NULL;

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
    Export ddm_cuda functions to python.
 */
void export_ddm_cuda(py::module &m)
{
    m.def("set_device", &set_device);
    // m.def("get_device_pitch", &get_device_pitch);
    // m.def("get_device_fft2_mem", &get_device_fft2_mem);
    // m.def("get_device_fft_mem", &get_device_fft_mem);
    // Leave function export in this order!
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint8_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<int16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<int32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<int64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<float>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<double>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<uint8_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<int16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<uint16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<int32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<uint32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<int64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<uint64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<float>, py::return_value_policy::take_ownership);
    m.def("ddm_fft_cuda", &ddm_fft_cuda<double>, py::return_value_policy::take_ownership);
}
