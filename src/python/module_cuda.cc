// SPDX-FileCopyrightText: 2023-present University of Vienna
// SPDX-FileCopyrightText: 2023-present Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino
// SPDX-License-Identifier: GPL-3.0-or-later

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
