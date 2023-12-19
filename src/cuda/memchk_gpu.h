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
    \param nx       number of grid points in x
    \param ny       number of grid points in y
    \param length   number of frames
    \param num_lags number of lags
*/
unsigned long long PYBIND11_EXPORT get_host_memory_diff(unsigned long long nx,
                                                        unsigned long long ny,
                                                        unsigned long long length,
                                                        unsigned long long num_lags);

/*! \brief Get the estimated RAM memory needed for the "fft" mode (in bytes)
    \param nx       number of grid points in x
    \param ny       number of grid points in y
    \param length   number of frames
    \param num_lags number of lags
*/
unsigned long long PYBIND11_EXPORT get_host_memory_fft(unsigned long long nx,
                                                       unsigned long long ny,
                                                       unsigned long long length,
                                                       unsigned long long num_lags);

/*! \brief Check if host memory is sufficient to execute the "diff" mode
    \param nx       number of grid points in x
    \param ny       number of grid points in y
    \param length   number of frames
    \param num_lags number of lags
    \return true if host memory is sufficient, false otherwise
*/
bool PYBIND11_EXPORT check_host_memory_diff(unsigned long long nx,
                                            unsigned long long ny,
                                            unsigned long long length,
                                            unsigned long long num_lags);

/*! \brief Check if host memory is sufficient to execute the "fft" mode
    \param nx       number of grid points in x
    \param ny       number of grid points in y
    \param length   number of frames
    \param num_lags number of lags
    \return true if host memory is sufficient, false otherwise
*/
bool PYBIND11_EXPORT check_host_memory_fft(unsigned long long nx,
                                           unsigned long long ny,
                                           unsigned long long length,
                                           unsigned long long num_lags);

#endif // __MEMCHK_GPU_H__
