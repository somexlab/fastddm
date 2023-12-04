// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __DDM_BINDINGS_H__
#define __DDM_BINDINGS_H__

/*! \file ddm_bindings.h
    \brief Declaration of bindings for C++ Differential Dynamic Microscopy functions
*/

// *** headers ***
#include "../../cpp/ddm.h"
#include "../../python_defs.h"

/*! \brief Export ddm functions to python
    \param m    Module
 */
void export_ddm(py::module &m)
{
    // Difference algorithm
    m.def("ddm_diff", &ddm_diff<uint8_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<int16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<uint16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<int32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<uint32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<int64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<uint64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<float>, py::return_value_policy::take_ownership);
    m.def("ddm_diff", &ddm_diff<double>, py::return_value_policy::take_ownership);
    // FFT optimized algorithm
    m.def("ddm_fft", &ddm_fft<uint8_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<int16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<uint16_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<int32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<uint32_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<int64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<uint64_t>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<float>, py::return_value_policy::take_ownership);
    m.def("ddm_fft", &ddm_fft<double>, py::return_value_policy::take_ownership);
}

#endif // __DDM_BINDINGS_H__
