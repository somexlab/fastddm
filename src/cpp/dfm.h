// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __DFM_H__
#define __DFM_H__

/*! \file dfm.h
    \brief Declaration of core C++ Digital Fourier Microscopy functions
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

/*! \brief Compute ISF in direct mode
    \param img_seq  numpy array containing the image sequence
    \param lags     lags to be analyzed
    \param nx       number of fft nodes in x direction
    \param ny       number of fft nodes in y direction
    \param logs     log messages
 */
template <typename T>
py::array_t<double> dfm_direct(py::array_t<T, py::array::c_style> img_seq,
                               vector<unsigned int> lags,
                               size_t nx,
                               size_t ny,
                               string &logs);

/*! \brief Compute ISF in fft mode using Wiener-Khinchin theorem
    \param img_seq      numpy array containing the image sequence
    \param lags         lags to be analyzed
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param nt           number of fft nodes in t direction
    \param bundle_size  number of fft's in the bundle
    \param logs         log messages
 */
template <typename T>
py::array_t<double> dfm_fft(py::array_t<T, py::array::c_style> img_seq,
                            vector<unsigned int> lags,
                            size_t nx,
                            size_t ny,
                            size_t nt,
                            size_t bundle_size,
                            string &logs);

/*! \brief Export dfm functions to python
    \param m    Module
 */
void export_dfm(py::module &m);

#endif