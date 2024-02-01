// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file module_cuda.cc
    \brief Export functions from CUDA to Python
*/

// *** headers ***
#include "bindings/ddm_cuda_bindings.h"

// *** code ***
/*
    Export with pybind11
 */
PYBIND11_MODULE(_core_cuda, m)
{
    export_ddm_cuda(m);
}
