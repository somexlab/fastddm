// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __DDM_CUDA_BINDINGS_H__
#define __DDM_CUDA_BINDINGS_H__

/*! \file ddm_cuda_bindings.h
    \brief Declaration of bindings for CUDA Differential Dynamic Microscopy functions
*/

// *** headers ***
#include "../../python_defs.h"
#include "../../cuda/gpu_utils.cuh"
#include "../../cuda/memchk_gpu.h"
#include "../../cuda/ddm_cuda.h"

/*! \brief Export ddm CUDA functions to Python
    \param m    Module
 */
void export_ddm_cuda(py::module &m)
{
    // Misc
    m.def("get_num_devices", &get_num_devices);
    m.def("set_device", &set_device);
    m.def("get_device", &get_device);
    m.def("get_free_device_memory", &get_free_device_memory);
    m.def("get_free_host_memory", &get_free_host_memory);
    m.def("get_host_memory_diff", &get_host_memory_diff);
    m.def("get_host_memory_fft", &get_host_memory_fft);
    m.def("check_host_memory_diff", &check_host_memory_diff_py);
    m.def("check_host_memory_fft", &check_host_memory_fft_py);
    // Difference algorithm
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint8_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<int16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<int32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<int64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<uint64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<float>, py::return_value_policy::take_ownership);
    m.def("ddm_diff_cuda", &ddm_diff_cuda<double>, py::return_value_policy::take_ownership);
    // FFT optimized algorithm
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

#endif // __DDM_CUDA_BINDINGS_H__
