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

/*! \brief Get the device memory pitch for multiple arrays of length N
    \param N    subarray size
 */
size_t get_device_pitch(size_t N);

/*! \brief Get the device memory for fft2
    \param nx   number of fft nodes in x direction
    \param ny   number of fft nodes in y direction
    \param nt   number of elements (in t direction)
 */
size_t get_device_fft2_mem(size_t nx,
                           size_t ny,
                           size_t nt);

/*! \brief Get the device memory for fft
    \param nt       number of fft nodes in t direction
    \param N        number of elements
    \param pitch    pitch of input array
 */
size_t get_device_fft_mem(size_t nt,
                          size_t N,
                          size_t pitch);

/*! \brief Export dfm cuda functions to python
    \param m    Module
 */
void export_dfm_cuda(py::module &m);

#endif