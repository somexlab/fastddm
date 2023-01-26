// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __DDM_H__
#define __DDM_H__

/*! \file ddm.h
    \brief Declaration of core C++ Differential Dynamic Microscopy functions
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

/*! \brief Compute image structure function in diff mode
    \param img_seq  numpy array containing the image sequence
    \param lags     lags to be analyzed
    \param nx       number of fft nodes in x direction
    \param ny       number of fft nodes in y direction
 */
template <typename T, typename S>
py::array_t<S> ddm_diff(py::array_t<T, py::array::c_style> img_seq,
                        vector<unsigned int> lags,
                        unsigned long long nx,
                        unsigned long long ny);

/*! \brief Compute image structure function in fft mode using Wiener-Khinchin theorem
    \param img_seq      numpy array containing the image sequence
    \param lags         lags to be analyzed
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param nt           number of fft nodes in t direction
    \param chunk_size   number of fft's in the chunk
 */
template <typename T, typename S>
py::array_t<S> ddm_fft(py::array_t<T, py::array::c_style> img_seq,
                       vector<unsigned int> lags,
                       unsigned long long nx,
                       unsigned long long ny,
                       unsigned long long nt,
                       unsigned long long chunk_size);

/*! \brief Export ddm functions to python
    \param m    Module
 */
void export_ddm(py::module &m);

#endif