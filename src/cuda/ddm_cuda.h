// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __DDM_CUDA_H__
#define __DDM_CUDA_H__

/*! \file ddm_cuda.h
    \brief Declaration of C++ handlers for Differential Dynamic Microscopy functions on GPU
*/

// *** headers ***
#include <vector>

#include "../python_defs.h"

using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Compute structure function in "diff" mode using differences on the GPU
    \param img_seq      Numpy array containing the image sequence
    \param lags         Lags to be analyzed
    \param nx           Number of grid points in x
    \param ny           Number of grid points in y
    \param window       Numpy array containing the window function to be applied to the images
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_diff_cuda(py::array_t<T, py::array::c_style> img_seq,
                                                  vector<unsigned int> lags,
                                                  unsigned long long nx,
                                                  unsigned long long ny,
                                                  py::array_t<Scalar, py::array::c_style> window);

/*! \brief Compute structure function in "fft" mode using Wiener-Khinchin theorem on the GPU
    \param img_seq      Numpy array containing the image sequence
    \param lags         Lags to be analyzed
    \param nx           Number of grid nodes in x
    \param ny           Number of grid nodes in y
    \param nt           Number of grid nodes in t
    \param window       Numpy array containing the window function to be applied to the images
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_fft_cuda(py::array_t<T, py::array::c_style> img_seq,
                                                 vector<unsigned int> lags,
                                                 unsigned long long nx,
                                                 unsigned long long ny,
                                                 unsigned long long nt,
                                                 py::array_t<Scalar, py::array::c_style> window);

#endif // __DDM_CUDA_H__
