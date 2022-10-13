// Maintainer: enrico-lattuada

/*! \file bindings.cc
    \brief Export functions from C++ to python
*/

// *** headers ***
#include "../cpp/dfm.h"

// *** code ***
/*
    Export with pybind11
 */
PYBIND11_MODULE(core, m) {

    export_dfm(m);
}