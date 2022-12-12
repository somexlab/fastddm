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
    \param N        subarray size
    \param Nbytes   element size in bytes
 */
size_t get_device_pitch(size_t N,
                        int Nbytes);

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

/*! \brief Compute ISF in direct mode on the GPU
    \param img_seq      numpy array containing the image sequence
    \param lags         lags to be analyzed
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param num_fft2     number of fft2 chunks
    \param buff_pitch   pitch of buffer device array
    \param num_chunks   number of q points chunks
    \param pitch_q      pitch of device array (q-pitch)
    \param pitch_t      pitch of device array (t-pitch)
 */
template <typename T>
py::array_t<double> dfm_direct_cuda(py::array_t<T, py::array::c_style> img_seq,
                                    vector<unsigned int> lags,
                                    size_t nx,
                                    size_t ny,
                                    size_t num_fft2,
                                    size_t buff_pitch,
                                    size_t num_chunks,
                                    size_t pitch_q,
                                    size_t pitch_t);

/*! \brief Export dfm cuda functions to python
    \param m    Module
 */
void export_dfm_cuda(py::module &m);

#endif