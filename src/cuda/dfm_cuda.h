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
    \param img_seq          numpy array containing the image sequence
    \param lags             lags to be analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param num_fft2         number of fft2 batches
    \param buff_pitch       pitch of buffer device array
    \param num_chunks       number of q points batches
    \param pitch_q          pitch of device array (q-pitch)
    \param pitch_t          pitch of device array (t-pitch)
    \param num_fullshift    number of full and shift batches
    \param pitch_fs         pitch of device array for full and shift operation
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
                                    size_t pitch_t,
                                    size_t num_fullshift,
                                    size_t pitch_fs);

/*! \brief Compute ISF in fft mode using Wiener-Khinchin theorem on the GPU
    \param img_seq          numpy array containing the image sequence
    \param lags             lags to be analyzed
    \param nx               number of fft nodes in x direction
    \param ny               number of fft nodes in y direction
    \param nt               number of fft nodes in t direction
    \param num_fft2         number of fft2 batches
    \param buff_pitch       pitch of buffer device array
    \param num_chunks       number of q points batches
    \param pitch_q          pitch of device array (q-pitch)
    \param pitch_t          pitch of device array (t-pitch)
    \param pitch_nt         pitch of workspace1 device array (nt-pitch)
    \param num_fullshift    number of full and shift batches
    \param pitch_fs         pitch of device array for full and shift operation
 */
template <typename T>
py::array_t<double> dfm_fft_cuda(py::array_t<T, py::array::c_style> img_seq,
                                 vector<unsigned int> lags,
                                 size_t nx,
                                 size_t ny,
                                 size_t nt,
                                 size_t num_fft2,
                                 size_t buff_pitch,
                                 size_t num_chunks,
                                 size_t pitch_q,
                                 size_t pitch_t,
                                 size_t pitch_nt,
                                 size_t num_fullshift,
                                 size_t pitch_fs);

/*! \brief Estimate and check host memory needed for direct mode
    \param mem_avail    Host memory available
    \param nx           Number of fft nodes, x direction
    \param ny           Number of fft nodes, y direction
    \param length       Number of frames
    \param lags         Vector of lags to analyze
 */
bool chk_host_mem_direct(unsigned long long mem_avail,
                         unsigned long long nx,
                         unsigned long long ny,
                         unsigned long long length,
                         vector<unsigned int> lags);

/*! \brief Estimate and check host memory needed for fft mode
    \param mem_avail    Host memory available
    \param nx           Number of fft nodes, x direction
    \param ny           Number of fft nodes, y direction
    \param length       Number of frames
    \param lags         Vector of lags to analyze
 */
bool chk_host_mem_fft(unsigned long long mem_avail,
                      unsigned long long nx,
                      unsigned long long ny,
                      unsigned long long length);

/*! \brief Export dfm cuda functions to python
    \param m    Module
 */
void export_dfm_cuda(py::module &m);

#endif