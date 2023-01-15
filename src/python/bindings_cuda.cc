// Maintainer: enrico-lattuada

/*! \file bindings_cuda.cc
    \brief Export cuda functions from C++ to python
*/

// *** headers ***
#include "../cuda/ddm_cuda.h"

// *** code ***
/*
    Export with pybind11
 */
PYBIND11_MODULE(core_cuda, m) {

    export_ddm_cuda(m);
}