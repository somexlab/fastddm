// Maintainer: enrico-lattuada

/*! \file bindings_cuda.cc
    \brief Export cuda functions from C++ to python
*/

// *** headers ***
#include "../cuda/dfm_cuda.h"

// *** code ***
/*
    Export with pybind11
 */
PYBIND11_MODULE(core_cuda, m) {

    export_dfm_cuda(m);
}