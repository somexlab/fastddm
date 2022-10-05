// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __PYIO_H__
#define __PYIO_H__

/*! \file pyio.h
    \brief Declaration of python/C++ communication functions
*/

// *** headers ***
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

// *** code ***

/*! \brief Copy python input to C++ vector
    \param buff buffer to numpy array
    \param dest destination vector
    \param nx   size along x after zero-padding
    \param ny   size along y after zero-padding
 */
void numpy2vector(const py::buffer_info buff,
                  vector<double> &dest,
                  size_t nx,
                  size_t ny);

#endif