// Maintainer: enrico-lattuada

/*! \file bindings.cc
    \brief Export functions from C++ to python
*/

// *** headers ***
#include "../cpp/ddm.h"
#include "../cuda/ddm_cuda.h"

// *** code ***
/*
    Export with pybind11
 */
PYBIND11_MODULE(_core, m) {

    export_ddm(m);
#if ENABLE_CUDA
    export_ddm_cuda(m);
#endif
}