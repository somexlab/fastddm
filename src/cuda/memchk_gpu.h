// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __MEMCHK_GPU_H__
#define __MEMCHK_GPU_H__

/*! \file memchk_gpu.h
    \brief Declaration of C++ functions for memory check and optimization for GPU routines
*/

// *** headers ***
#include "../python_defs.h"

#include "data_struct.h"

using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Get free host memory (in bytes)
    \return free host memory (in bytes)
*/
unsigned long long PYBIND11_EXPORT get_free_host_memory();

/*! \brief Get the estimated RAM memory needed for the "diff" mode (in bytes)
    \param nx       Number of grid points in x
    \param ny       Number of grid points in y
    \param length   Number of frames
    \param num_lags Number of lags
*/
unsigned long long PYBIND11_EXPORT get_host_memory_diff(unsigned long long nx,
                                                        unsigned long long ny,
                                                        unsigned long long length,
                                                        unsigned long long num_lags);

/*! \brief Get the estimated RAM memory needed for the "fft" mode (in bytes)
    \param nx       Number of grid points in x
    \param ny       Number of grid points in y
    \param length   Number of frames
    \param num_lags Number of lags
*/
unsigned long long PYBIND11_EXPORT get_host_memory_fft(unsigned long long nx,
                                                       unsigned long long ny,
                                                       unsigned long long length,
                                                       unsigned long long num_lags);

/*! \brief Check if host memory is sufficient to execute the "diff" mode (Python interface)
    \param nx       Number of grid points in x
    \param ny       Number of grid points in y
    \param length   Number of frames
    \param num_lags Number of lags
    \return true if host memory is sufficient, false otherwise
*/
bool PYBIND11_EXPORT check_host_memory_diff_py(unsigned long long nx,
                                               unsigned long long ny,
                                               unsigned long long length,
                                               unsigned long long num_lags);

/*! \brief Check if host memory is sufficient to execute the "diff" mode
    \param img_data      Structure holding the image sequence parameters
    \param sf_data       Structure holding the structure function parameters
    \return true if host memory is sufficient, false otherwise
*/
bool check_host_memory_diff(ImageData &img_data,
                            StructureFunctionData &sf_data);

/*! \brief Check if host memory is sufficient to execute the "fft" mode (Python interface)
    \param nx       Number of grid points in x
    \param ny       Number of grid points in y
    \param length   Number of frames
    \param num_lags Number of lags
    \return true if host memory is sufficient, false otherwise
*/
bool PYBIND11_EXPORT check_host_memory_fft_py(unsigned long long nx,
                                              unsigned long long ny,
                                              unsigned long long length,
                                              unsigned long long num_lags);

/*! \brief Check if host memory is sufficient to execute the "fft" mode
    \param img_data      Structure holding the image sequence parameters
    \param sf_data       Structure holding the structure function parameters
    \return true if host memory is sufficient, false otherwise
*/
bool check_host_memory_fft(ImageData &img_data,
                           StructureFunctionData &sf_data);

#endif // __MEMCHK_GPU_H__
