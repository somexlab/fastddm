// SPDX-FileCopyrightText: 2023-present University of Vienna
// SPDX-License-Identifier: GPL-3.0-or-later

/*! \file module.cc
    \brief Export functions from C++ to Python
*/

// *** headers ***
#include "bindings/ddm_bindings.h"

// *** code ***
/*
    Export with pybind11
 */
PYBIND11_MODULE(_core, m)
{
    export_ddm(m);
}
