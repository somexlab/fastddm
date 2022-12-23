// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __DFM_CUDA_H__
#define __DFM_CUDA_H__

/*! \file dfm_cuda.h
    \brief Declaration of C++ handlers for Digital Fourier Microscopy functions on GPU
*/

// *** headers ***
#include <vector>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
using namespace std;

// *** code ***

/*! \brief Compute ISF in direct mode on the GPU
    \param img_seq          numpy array containing the image sequence
    \param lags             lags to be analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
 */
template <typename T>
py::array_t<double> dfm_direct_cuda(py::array_t<T, py::array::c_style> img_seq,
                                    vector<unsigned int> lags,
                                    size_t nx,
                                    size_t ny);

/*! \brief Compute ISF in fft mode using Wiener-Khinchin theorem on the GPU
    \param img_seq          numpy array containing the image sequence
    \param lags             lags to be analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param nt               number of fft nodes in t direction
 */
template <typename T>
py::array_t<double> dfm_fft_cuda(py::array_t<T, py::array::c_style> img_seq,
                                 vector<unsigned int> lags,
                                 size_t nx,
                                 size_t ny,
                                 size_t nt);

/*! \brief Set CUDA device to be used
    \param gpu_id   The device id (starts from 0)
*/
void set_device(int gpu_id);

/*! \brief Export dfm cuda functions to python
    \param m    Module
 */
void export_dfm_cuda(py::module &m);

#endif