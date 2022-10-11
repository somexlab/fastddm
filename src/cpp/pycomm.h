// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __PYCOMM_H__
#define __PYCOMM_H__

/*! \file pycomm.h
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

/*! \brief Return C++ vector as numpy array
    \param vec  ponter to vector
    \param nx   output size along x
    \param ny   output size along y
    \param nt   output size along t
 */
py::array_t<double> vector2numpy(vector<double> *vec,
                                 size_t nx,
                                 size_t ny,
                                 size_t nt);

#endif