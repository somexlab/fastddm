// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file ddm_cuda.cc
    \brief Definition of C++ handlers for Differential Dynamic Microscopy functions on the GPU
*/

// *** headers ***
#include "ddm_cuda.h"
#include "memchk_gpu.h"
#include "data_struct.h"

#include "ddm_cuda.cuh"
#include "memchk_gpu.cuh"

// *** code ***

/*!
    Compute the structure function in "diff" mode
    using differences of Fourier transformed images on the GPU.
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<T, py::array::c_style> img_seq,
                                                  vector<unsigned int> lags,
                                                  unsigned long long nx,
                                                  unsigned long long ny,
                                                  py::array_t<Scalar, py::array::c_style> window)
{
    // Get buffer info to image sequence and window
    py::buffer_info img_seq_info = img_seq.request();
    py::buffer_info window_info = window.request();

    // Get image sequence array and dimensions
    T *img_seq_ptr = static_cast<T *>(img_seq_info.ptr);

    // Get image sequence parameters
    ImageData img_data;
    img_data.length = img_seq_info.shape[0];
    img_data.height = img_seq_info.shape[1];
    img_data.width = img_seq_info.shape[2];
    img_data.is_input_type_scalar = std::is_same<T, Scalar>::value;
    img_data.input_type_num_bytes = sizeof(T);

    // Get window array
    Scalar *window_ptr = static_cast<Scalar *>(window_info.ptr);

    // Get window parameters
    StructureFunctionData sf_data;
    sf_data.nx = nx;
    sf_data.ny = ny;
    sf_data.num_lags = lags.size();
    sf_data.length = lags.size() + 2ULL;
    sf_data.nx_half = nx / 2ULL + 1ULL;
    // We check if the window is empty to understand if a window is applied or not
    unsigned long long window_length = window_info.shape[0];
    sf_data.is_window = window_length > 0;

    // Check host memory
    bool is_mem_ok = check_host_memory_diff(img_data, sf_data);
    if (!is_mem_ok)
    {
        throw std::runtime_error("Not enough space in memory to store the result.\n");
    }

    // Check device memory and optimize kernel execution
    ExecutionParameters exec_params;
    PitchData pitch_data;
    check_and_optimize_device_memory_diff(img_data,
                                          sf_data,
                                          exec_params,
                                          pitch_data);

    // Allocate workspace memory
    /*
        We need to make sure that the fft2 r2c fits in the array, so the size of one fft2 output is
        [ny * (nx // 2 + 1)]   complex Scalar
        The workspace array (of Scalar) needs to be twice as large.

        We also need to make sure that the full output fits in the array, also including the average
        power spectrum of the input images and the variance of their spatial FFT2 outputs.
        The workspace array must be as long as the maximum of the length of the input image sequence
        and the number of lags +2.
     */
    // Compute the length of the workspace
    unsigned long long dim_t = max(img_data.length, sf_data.length);
    // Create the output array and get the buffer info
    py::array_t<Scalar> result = py::array_t<Scalar>(dim_t * sf_data.ny * sf_data.nx_half * 2ULL);
    py::buffer_info result_info = result.request();

    // Get pointer to the output array
    Scalar *result_ptr = static_cast<Scalar *>(result_info.ptr);

    // Compute the FFT2 on the GPU
    compute_fft2(img_seq_ptr,
                 result_ptr,
                 window_ptr,
                 img_data,
                 sf_data,
                 exec_params,
                 pitch_data);

    // Compute the structure function on the GPU
    structure_function_diff(result_ptr,
                            lags,
                            img_data,
                            sf_data,
                            exec_params,
                            pitch_data);

    // Convert raw output to shifted structure function
    make_shift(result_ptr,
               img_data,
               sf_data,
               exec_params,
               pitch_data);

    // Reshape and resize the output array
    // The full size of the structure function is
    // nx * ny * (#lags + 2)
    result.resize({sf_data.length, sf_data.ny, sf_data.nx_half});

    // Return the output
    return result;
}
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<uint8_t, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<int16_t, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<uint16_t, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<int32_t, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<uint32_t, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<int64_t, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<uint64_t, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<float, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<double, py::array::c_style> img_seq,
                                                           vector<unsigned int> lags,
                                                           unsigned long long nx,
                                                           unsigned long long ny,
                                                           py::array_t<Scalar, py::array::c_style> window);

/*!
    Compute the structure function in "fft" mode
    using the Wiener-Khinchin theorem on the GPU.

    Notice that nt must be at least 2*length to avoid
    circular correlation.
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<T, py::array::c_style> img_seq,
                                                 vector<unsigned int> lags,
                                                 unsigned long long nx,
                                                 unsigned long long ny,
                                                 unsigned long long nt,
                                                 py::array_t<Scalar, py::array::c_style> window)
{
    // Get buffer info to image sequence and window
    py::buffer_info img_seq_info = img_seq.request();
    py::buffer_info window_info = window.request();

    // Get image sequence array and dimensions
    T *img_seq_ptr = static_cast<T *>(img_seq_info.ptr);

    // Get image sequence parameters
    ImageData img_data;
    img_data.length = img_seq_info.shape[0];
    img_data.height = img_seq_info.shape[1];
    img_data.width = img_seq_info.shape[2];
    img_data.is_input_type_scalar = std::is_same<T, Scalar>::value;
    img_data.input_type_num_bytes = sizeof(T);

    // Get window array
    Scalar *window_ptr = static_cast<Scalar *>(window_info.ptr);

    // Get window parameters
    StructureFunctionData sf_data;
    sf_data.nx = nx;
    sf_data.ny = ny;
    sf_data.num_lags = lags.size();
    sf_data.length = lags.size() + 2ULL;
    sf_data.nx_half = nx / 2ULL + 1ULL;
    // We check if the window is empty to understand if a window is applied or not
    unsigned long long window_length = window_info.shape[0];
    sf_data.is_window = window_length > 0;

    // Check host memory
    bool is_mem_ok = check_host_memory_fft(img_data, sf_data);
    if (!is_mem_ok)
    {
        throw std::runtime_error("Not enough space in memory to store the result.\n");
    }

    // Check device memory and optimize kernel execution
    ExecutionParameters exec_params;
    PitchData pitch_data;
    check_and_optimize_device_memory_fft(nt,
                                         img_data,
                                         sf_data,
                                         exec_params,
                                         pitch_data);

    // Allocate workspace memory
    /*
        We need to make sure that the fft2 r2c fits in the array, so the size of one fft2 output is
        [ny * (nx // 2 + 1)]   complex Scalar
        The workspace array (of Scalar) needs to be twice as large.

        We also need to make sure that the full output fits in the array, also including the average
        power spectrum of the input images and the variance of their spatial FFT2 outputs.
        The workspace array must be as long as the maximum of the length of the input image sequence
        and the number of lags +2.
     */
    // Compute the length of the workspace
    unsigned long long dim_t = max(img_data.length, sf_data.length);
    // Create the output array and get the buffer info
    py::array_t<Scalar> result = py::array_t<Scalar>(dim_t * sf_data.ny * sf_data.nx_half * 2ULL);
    py::buffer_info result_info = result.request();

    // Get pointer to the output array
    Scalar *result_ptr = static_cast<Scalar *>(result_info.ptr);

    // Compute the FFT2 on the GPU
    compute_fft2(img_seq_ptr,
                 result_ptr,
                 window_ptr,
                 img_data,
                 sf_data,
                 exec_params,
                 pitch_data);

    // Compute the structure function on the GPU
    structure_function_fft(result_ptr,
                           lags,
                           img_data,
                           sf_data,
                           exec_params,
                           pitch_data);

    // Convert raw output to shifted structure function
    make_shift(result_ptr,
               img_data,
               sf_data,
               exec_params,
               pitch_data);

    // Reshape and resize the output array
    // The full size of the structure function is
    // nx * ny * (#lags + 2)
    result.resize({sf_data.length, sf_data.ny, sf_data.nx_half});

    // Return the output
    return result;
}
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<uint8_t, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<int16_t, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<uint16_t, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<int32_t, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<uint32_t, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<int64_t, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<uint64_t, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<float, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<double, py::array::c_style> img_seq,
                                                          vector<unsigned int> lags,
                                                          unsigned long long nx,
                                                          unsigned long long ny,
                                                          unsigned long long nt,
                                                          py::array_t<Scalar, py::array::c_style> window);
