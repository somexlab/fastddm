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
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Compute image structure function in diff mode using differences on the GPU
    \param img_seq      numpy array containing the image sequence
    \param lags         lags to be analyzed
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param window       numpy array containing the window function to be applied to the images
 */
template <typename T>
py::array_t<Scalar> ddm_diff_cuda(py::array_t<T, py::array::c_style> img_seq,
                                  vector<unsigned int> lags,
                                  unsigned long long nx,
                                  unsigned long long ny,
                                  py::array_t<Scalar, py::array::c_style> window);

/*! \brief Compute image structure function in fft mode using Wiener-Khinchin theorem on the GPU
    \param img_seq      numpy array containing the image sequence
    \param lags         lags to be analyzed
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param nt           number of fft nodes in t direction
    \param window       numpy array containing the window function to be applied to the images
 */
template <typename T>
py::array_t<Scalar> ddm_fft_cuda(py::array_t<T, py::array::c_style> img_seq,
                                 vector<unsigned int> lags,
                                 unsigned long long nx,
                                 unsigned long long ny,
                                 unsigned long long nt,
                                 py::array_t<Scalar, py::array::c_style> window);

/*! \brief Set CUDA device to be used
    \param gpu_id       The device id (starts from 0)
*/
void set_device(int gpu_id);

/*! \brief Export ddm cuda functions to python
    \param m    Module
 */
void export_ddm_cuda(py::module &m);

#endif // __DDM_CUDA_H__
