// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file memchk_gpu.cc
    \brief Definition of C++ functions for memory check and optimization for GPU routines
*/

// *** headers ***
#include "memchk_gpu.h"

// system includes
#ifdef WIN32
#include <windows.h>
#elif APPLE
#include <sys/sysinfo.h>
#else
#include <fstream>
#endif

// define scalar size
#ifndef SINGLE_PRECISION
unsigned long long SCALAR_SIZE = 8;
#else
unsigned long long SCALAR_SIZE = 4;
#endif

// *** code ***

/*!
    Get free host memory (in bytes)
 */
unsigned long long PYBIND11_EXPORT get_free_host_memory()
{
    unsigned long long free_mem;

#ifdef WIN32

    MEMORYSTATUSEX memoryStatus;
    memoryStatus.dwLength = sizeof(memoryStatus);
    if (!GlobalMemoryStatusEx(&memoryStatus))
    {
        throw std::runtime_error("Failed to retrieve memory status.\n");
        return 0;
    }

    free_mem = memoryStatus.ullAvailPhys;

#elif APPLE // APPLE not supported for CUDA core at the moment...

    struct sysinfo info;
    if (sysinfo(&info) != 0)
    {
        throw std::runtime_error("Failed to retrieve system information.\n");
        return 0;
    }

    free_mem = info.freeram + info.bufferram;

#else

    // https://github.com/doleron/cpp-linux-system-stats/blob/main/include/linux-system-usage.hpp
    ifstream proc_meminfo("/proc/meminfo");

    if (!proc_meminfo.good())
    {
        throw std::runtime_error("Failed to retrieve memory information.\n");
        return 0;
    }

    string content((istreambuf_iterator<char>(proc_meminfo)), istreambuf_iterator<char>());
    string target = "MemAvailable:";
    size_t start = content.find(target);
    if (start != string::npos)
    {
        int begin = start + target.length();
        size_t end = content.find("kB", start);
        string substr = content.substr(begin, end - begin);
        free_mem = stoull(substr) * 1024;
    }

#endif

    return free_mem;
}

/*!
    Get the estimated RAM memory needed for the "diff" mode (in bytes)
*/
unsigned long long PYBIND11_EXPORT get_host_memory_diff(unsigned long long nx,
                                                        unsigned long long ny,
                                                        unsigned long long length,
                                                        unsigned long long num_lags)
{
    /*
    Calculations are done in single or double precision.
    - The store the output, we need
        (nx / 2 + 1) * ny * (num_lags + 2) * (SCALAR_SIZE) bytes

    - To store the fft2, we need
        (nx / 2 + 1) * ny * length * 2 * (SCALAR_SIZE) bytes

    To store both, we need
        (nx / 2 + 1) * ny * max(length * 2, num_lags + 2) * (SCALAR_SIZE) bytes
     */
    unsigned long long mem_required = 0;

    unsigned long long dim_t = max(length * 2, num_lags + 2);
    mem_required += (nx / 2ULL + 1ULL) * ny * dim_t * SCALAR_SIZE;

    return mem_required;
}

/*!
    Get the estimated RAM memory needed for the "fft" mode (in bytes)
*/
unsigned long long PYBIND11_EXPORT get_host_memory_fft(unsigned long long nx,
                                                       unsigned long long ny,
                                                       unsigned long long length,
                                                       unsigned long long num_lags)
{
    /*
    Calculations are done in single or double precision.
    - The store the output, we need
        (nx / 2 + 1) * ny * (num_lags + 2) * (SCALAR_SIZE) bytes

    - To store the fft2, we need
        (nx / 2 + 1) * ny * length * 2 * (SCALAR_SIZE) bytes

    To store both, we need
        (nx / 2 + 1) * ny * max(length * 2, num_lags + 2) * (SCALAR_SIZE) bytes
     */

    // The size required on host is the same as the size required for the "diff" mode
    // Return it to avoid code duplication
    return get_host_memory_diff(nx, ny, length, num_lags);
}

/*!
    Check if host memory is sufficient to execute the "diff" mode (Python interface)
*/
bool PYBIND11_EXPORT check_host_memory_diff_py(unsigned long long nx,
                                               unsigned long long ny,
                                               unsigned long long length,
                                               unsigned long long num_lags)
{
    // Get the free host memory and estimated memory needed for the "diff" mode
    unsigned long long free_mem = get_free_host_memory();
    unsigned long long mem_required = get_host_memory_diff(nx, ny, length, num_lags);

    // Scale the free memory by 0.95 to leave some free space
    free_mem = (unsigned long long)(0.95 * (double)free_mem);

    return (free_mem > mem_required);
}

/*!
    Check if host memory is sufficient to execute the "diff" mode
*/
bool check_host_memory_diff(ImageData &img_data,
                            StructureFunctionData &sf_data)
{
    return check_host_memory_diff_py(sf_data.nx,
                                     sf_data.ny,
                                     img_data.length,
                                     sf_data.num_lags);
}

/*!
    Check if host memory is sufficient to execute the "fft" mode (Python interface)
*/
bool PYBIND11_EXPORT check_host_memory_fft_py(unsigned long long nx,
                                              unsigned long long ny,
                                              unsigned long long length,
                                              unsigned long long num_lags)
{
    // Get the free host memory and estimated memory needed for the "fft" mode
    unsigned long long free_mem = get_free_host_memory();
    unsigned long long mem_required = get_host_memory_fft(nx, ny, length, num_lags);

    // Scale the free memory by 0.95 to leave some free space
    free_mem = (unsigned long long)(0.95 * (double)free_mem);

    return (free_mem > mem_required);
}

/*!
    Check if host memory is sufficient to execute the "fft" mode
*/
bool check_host_memory_fft(ImageData &img_data,
                           StructureFunctionData &sf_data)
{
    return check_host_memory_fft_py(sf_data.nx,
                                    sf_data.ny,
                                    img_data.length,
                                    sf_data.num_lags);
}
