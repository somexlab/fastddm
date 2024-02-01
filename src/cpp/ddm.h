// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __DDM_H__
#define __DDM_H__

/*! \file ddm.h
    \brief Declaration of core C++ Differential Dynamic Microscopy functions
*/

// *** headers ***
#include <vector>
#include <string>

#include "../python_defs.h"

using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Compute structure function in "diff" mode
    \param img_seq  Numpy array containing the image sequence
    \param lags     Lags to be analyzed
    \param nx       Number of fft nodes in x direction
    \param ny       Number of fft nodes in y direction
    \param window   Numpy array containing the window function to be applied to the images
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<T, py::array::c_style> img_seq,
                                             vector<unsigned int> lags,
                                             unsigned long long nx,
                                             unsigned long long ny,
                                             py::array_t<Scalar, py::array::c_style> window);

/*! \brief Compute structure function in "fft" mode using Wiener-Khinchin theorem
    \param img_seq      Numpy array containing the image sequence
    \param lags         Lags to be analyzed
    \param nx           Number of fft nodes in x direction
    \param ny           Number of fft nodes in y direction
    \param nt           Number of fft nodes in t direction
    \param chunk_size   Number of fft's in the chunk
    \param window       Numpy array containing the window function to be applied to the images
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<T, py::array::c_style> img_seq,
                                            vector<unsigned int> lags,
                                            unsigned long long nx,
                                            unsigned long long ny,
                                            unsigned long long nt,
                                            unsigned long long chunk_size,
                                            py::array_t<Scalar, py::array::c_style> window);

#endif // __DDM_H__
