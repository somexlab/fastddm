// Maintainer: enrico-lattuada

/*! \file pycomm.cc
    \brief Definition of python/C++ communication functions
*/

// *** headers ***
#include "pycomm.h"

// *** code ***

/*!
    Return a C++ vector as a python numpy array.
 */
py::array_t<double> vector2numpy(vector<double> *vec,
                                 size_t nx,
                                 size_t ny,
                                 size_t nt)
{
    // This snippet is to avoid copy when return vector as numpy array
    py::capsule free_when_done(vec, [](void *f)
                               {
      auto foo = reinterpret_cast<std::vector<double> *>(f);
      delete foo; });

    size_t stride_t = nx * ny * sizeof(double);
    size_t stride_y = nx * sizeof(double);
    size_t stride_x = sizeof(double);

    return py::array_t<double>({nt, ny, nx},                   // shape
                               {stride_t, stride_y, stride_x}, // C-style contiguous strides for double
                               vec->data(),                    // data pointer
                               free_when_done);
}