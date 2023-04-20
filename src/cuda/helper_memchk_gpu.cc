// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file helper_memchk_gpu.cc
    \brief Definition of C++ helper functions for memory check and optimization for GPU routines
*/

// *** headers ***
#include "helper_memchk_gpu.h"
#include "helper_memchk_gpu.cuh"

#include <cuda_runtime.h>
#include <cuda.h>
#include <nvml.h>

#ifdef WIN32
#include <windows.h>
#elif APPLE
#include <sys/sysinfo.h>
#else
#include <fstream>
#endif

#include <stdexcept>

#ifndef SINGLE_PRECISION
unsigned long long SCALAR_SIZE = 8;
#else
unsigned long long SCALAR_SIZE = 4;
#endif

// *** code ***

/*!
    Get free host memory (in bytes)
 */
void get_host_free_mem(unsigned long long &free_mem)
{
#ifdef WIN32
    MEMORYSTATUSEX statex;
    GlobalMemoryStatusEx(&statex);
    free_mem = statex.ullAvailPhys;
#elif APPLE
    struct sysinfo info;
    sysinfo(&info);
    free_mem = info.freeram + info.bufferram;
#else
    // https://github.com/doleron/cpp-linux-system-stats/blob/main/include/linux-system-usage.hpp
    ifstream proc_meminfo("/proc/meminfo");

    if (proc_meminfo.good())
    {
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
    }
#endif
}

/*!
    Estimate and check host memory needed for diff mode
 */
void chk_host_mem_diff(unsigned long long nx,
                       unsigned long long ny,
                       unsigned long long length,
                       unsigned long long Nlags)
{
    unsigned long long free_mem;
    get_host_free_mem(free_mem);

    // leave some free space
    free_mem = (unsigned long long)(0.95 * (double)free_mem);

    /*
    Calculations are done in single or double precision.
    - The store the output, we need
        (nx / 2 + 1) * ny * (#lags + 2) * (SCALAR_SIZE) bytes

    - To store the fft2, we need
        (nx / 2 + 1) * ny * length * 2 * (SCALAR_SIZE) bytes

    To store both, we need
        (nx / 2 + 1) * ny * max(2 * length, #lags + 2) * (SCALAR_SIZE) bytes
     */
    unsigned long long mem_required = 0;

    unsigned long long dim_t = max(2 * length, Nlags + 2);
    mem_required += (nx / 2ULL + 1ULL) * ny * dim_t * SCALAR_SIZE;

    if (mem_required >= free_mem)
    {
        throw std::runtime_error("Not enough space. Cannot store result in memory.");
    }
}

/*!
    Estimate and check host memory needed for fft mode
 */
void chk_host_mem_fft(unsigned long long nx,
                      unsigned long long ny,
                      unsigned long long length,
                      unsigned long long Nlags)
{
    unsigned long long free_mem;
    get_host_free_mem(free_mem);

    // leave some free space
    free_mem = (unsigned long long)(0.95 * (double)free_mem);

    /*
    Calculations are done in single or double precision.
    - The store the output, we need
        (nx / 2 + 1) * ny * (#lags + 2) * (SCALAR_SIZE) bytes

    - To store the fft2, we need
        (nx / 2 + 1) * ny * length * 2 * (SCALAR_SIZE) bytes

    To store both, we need
        (nx / 2 + 1) * ny * max(2 * length, #lags + 2) * (SCALAR_SIZE) bytes
     */
    unsigned long long mem_required = 0;

    unsigned long long dim_t = max(2 * length, Nlags + 2);
    mem_required += (nx / 2ULL + 1ULL) * ny * dim_t * SCALAR_SIZE;

    if (mem_required >= free_mem)
    {
        throw std::runtime_error("Not enough space. Cannot store result in memory.");
    }
}

/*!
    Get free device memory (in bytes)
 */
void get_device_free_mem(unsigned long long &free_mem)
{
    int device_id;
    cudaGetDevice(&device_id);

    // get device memory
    nvmlInit_v2();
    nvmlDevice_t dev;
    nvmlReturn_t res = nvmlDeviceGetHandleByIndex_v2((unsigned int)device_id, &dev);
    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(dev, &mem);
    nvmlShutdown();

    free_mem = mem.free;
}

/*!
    Evaluate the device memory pitch for multiple subarrays of size N
 */
unsigned long long get_device_pitch(unsigned long long N,
                                    int Nbytes)
{
    size_t pitch;
    switch (Nbytes)
    {
    case 16:
        cudaGetDevicePitch16B(N, pitch);
        break;
    case 8:
        cudaGetDevicePitch8B(N, pitch);
        break;
    case 4:
        cudaGetDevicePitch4B(N, pitch);
        break;
    case 2:
        cudaGetDevicePitch2B(N, pitch);
        break;
    case 1:
        cudaGetDevicePitch1B(N, pitch);
        break;
    default:
        cudaGetDevicePitch8B(N, pitch);
    }

    return (unsigned long long)pitch;
}

/*!
    Get the device memory for fft2
 */
unsigned long long get_device_fft2_mem(unsigned long long nx,
                                       unsigned long long ny,
                                       unsigned long long batch,
                                       unsigned long long pitch,
                                       cufftResult &cufft_res)
{
    // The following line should be changed to move to workstations with multiple GPUs
    size_t memsize[1]; // We are only considering workstations with 1 GPU
    cudaGetFft2MemSize(nx,
                       ny,
                       batch,
                       pitch,
                       memsize,
                       cufft_res);

    return (unsigned long long)memsize[0];
}

/*!
    Get the device memory for fft
 */
unsigned long long get_device_fft_mem(unsigned long long nt,
                                      unsigned long long batch,
                                      unsigned long long pitch,
                                      cufftResult &cufft_res)
{
    // The following line should be changed to move to workstations with multiple GPUs
    size_t memsize[1]; // We are only considering workstations with 1 GPU
    cudaGetFftMemSize(nt,
                      batch,
                      pitch,
                      memsize,
                      cufft_res);

    return (unsigned long long)memsize[0];
}

/*!
    Optimize fft2 execution parameters based on available gpu memory.

    Writes in the corresponding arguments:
        - the number of iterations for fft2 (frame chunks)
        - the pitch in number of elements for buffer array (real values)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
 */
void optimize_fft2(unsigned long long &pitch_buff,
                   unsigned long long &pitch_nx,
                   unsigned long long &num_fft2,
                   bool is_input_Scalar,
                   unsigned long long pixel_Nbytes,
                   unsigned long long width,
                   unsigned long long height,
                   unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long free_mem)
{
    // Calculations are done in double precision.
    // But data are transferred as Scalar (float/double)
    // To compute the fft2, we need
    //  - for the buffer (only if input is not Scalar):
    //      pitch_nx * height * fft2_batch_len * pixel_Nbytes bytes
    //  - for the workspace (complex double, 16 bytes):
    //      (nx / 2 + 1) * ny * fft2_batch_len * 16 bytes
    //  - for the cufft2 internal buffer:
    //      {programmatically determined...}

    unsigned long long _nx = nx / 2ULL + 1ULL;

    // memory required
    unsigned long long mem_req = 0ULL;

    // get device pitch for buffer array (only if input is not Scalar)
    pitch_buff = is_input_Scalar ? 0ULL : get_device_pitch(width, pixel_Nbytes);

    // get device pitch for fft2 output complex array
    pitch_nx = get_device_pitch(_nx, 2 * sizeof(Scalar));

    // start with worst case scenario:
    // we need to perform as many fft2 loops as the number of images
    num_fft2 = length;
    unsigned long long prev_num_fft2;

    while (true)
    {
        // reset memory required
        mem_req = 0ULL;

        // compute the number of batched fft2
        unsigned long long fft2_batch_len = (length + num_fft2 - 1ULL) / num_fft2;

        // estimate cufft2 internal buffer size
        cufftResult res;
        unsigned long long mem_fft2 = get_device_fft2_mem(nx, ny, fft2_batch_len, pitch_nx, res);
        if (res == CUFFT_SUCCESS)
        {
            // add internal buffer memory required for cufft2
            mem_req += mem_fft2;

            // add buffer memory required for input images (only if they are not already Scalar)
            mem_req += pitch_buff * height * fft2_batch_len * (unsigned long long)pixel_Nbytes;

            // add memory required for workspace
            mem_req += pitch_nx * ny * fft2_batch_len * 16ULL;

            // check memory
            if (free_mem > mem_req)
            {
                // estimate new num_fft2
                unsigned long long new_num_fft2 = (num_fft2 * mem_req + free_mem - 1ULL) / free_mem;
                if (new_num_fft2 == prev_num_fft2)
                {
                    break;
                }
                else
                {
                    prev_num_fft2 = num_fft2;
                    num_fft2 = new_num_fft2;
                }
            }
            else if (num_fft2 == length)
            {
                throw std::runtime_error("Not enough space on GPU for fft2.");
            }
            else
            {
                num_fft2 = prev_num_fft2;
                break;
            }
        }
        else if (num_fft2 == length)
        {
            throw std::runtime_error("Not enough space on GPU for fft2. cufftResult #: " + res);
        }
    }
}

/*!
    Optimize shift execution parameters based on available gpu memory.

    Writes in the corresponding arguments:
        - the number of iterations for fftshift (frame chunks)
        - the pitch in number of elements for shift workspace (pitch_fs, complex Scalar)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
 */
void optimize_shift(unsigned long long &pitch_fs,
                    unsigned long long &num_shift,
                    unsigned long long nx,
                    unsigned long long ny,
                    unsigned long long num_lags,
                    unsigned long long free_mem)
{
    // Calculations are done in single or double precision.
    // To compute the shift conversion, we need
    //  - workspace1 (complex Scalar, 2 * SCALAR_SIZE bytes)
    //      pitch_fs * ny * shift_batch_len * 2 * (SCALAR_SIZE) bytes
    //  - workspace2 (double, SCALAR_SIZE bytes)
    //      pitch_fs * ny * shift_batch_len * (SCALAR_SIZE) bytes

    // memory required
    unsigned long long mem_req = 0ULL;

    // get device pitch for workspace array (shift pitch, complex Scalar)
    pitch_fs = get_device_pitch((nx / 2ULL + 1ULL), 2 * SCALAR_SIZE);

    // start with worst case scenario:
    // we need to perform as many shift loops as the number of frames (num_lags)
    num_shift = num_lags;
    unsigned long long prev_num_shift;

    while (true)
    {
        // reset memory required
        mem_req = 0ULL;

        // compute the number of batched q vectors
        unsigned long long shift_batch_len = (num_lags + num_shift - 1ULL) / num_shift;

        // add workspace1 and workspace2 memory
        mem_req += pitch_fs * ny * shift_batch_len * 2ULL * SCALAR_SIZE;
        mem_req += pitch_fs * ny * shift_batch_len * SCALAR_SIZE;

        // check memory
        if (free_mem > mem_req)
        {
            // estimate new num_shift
            unsigned long long new_num_shift = (num_shift * mem_req + free_mem - 1ULL) / free_mem;
            if (new_num_shift == prev_num_shift)
            {
                break;
            }
            else
            {
                prev_num_shift = num_shift;
                num_shift = new_num_shift;
            }
        }
        else if (num_shift == num_lags)
        {
            throw std::runtime_error("Not enough space on GPU for shifted power spectrum.");
        }
        else
        {
            num_shift = prev_num_shift;
            break;
        }
    }
}

/*!
    Optimize structure function diff execution parameters based on available gpu memory

    Writes in the corresponding arguments:
        - the number of iterations for structure function (q-vector chunks)
        - the pitch in number of elements for workspace (pitch_q, complex double)
        - the pitch in number of elements for workspace (pitch_t, complex double)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
 */
void optimize_diff(unsigned long long &pitch_q,
                   unsigned long long &pitch_t,
                   unsigned long long &num_chunks,
                   unsigned long long length,
                   unsigned long long nx,
                   unsigned long long ny,
                   unsigned long long num_lags,
                   unsigned long long free_mem)
{
    // Calculations are done in double precision.
    // But data are transferred in single or double precision.
    // To compute the image structure function in diff mode, we need
    //  - helper lags array (unsigned int, 4 bytes)
    //      lags.size() * 4 bytes
    //  - workspace1 and workspace2 (complex double, 16 bytes)
    //      max(pitch_q * length, chunk_size * pitch_t) * 16 bytes
    //  - power_spec and var arrays (double2, 16 bytes)
    //      2 * chunk_size * 16 bytes

    // memory required
    unsigned long long mem_req = 0ULL;

    // get device pitch for workspace array (time pitch, complex double)
    pitch_t = get_device_pitch(length, 16);

    // start with worst case scenario:
    // we need to perform as many loop iterations as the number of q vectors
    num_chunks = (nx / 2ULL + 1ULL) * ny;
    unsigned long long prev_num_chunks;

    while (true)
    {
        // reset memory required
        mem_req = 0ULL;

        // compute the number of batched q vectors
        unsigned long long chunk_size = ((nx / 2ULL + 1ULL) * ny + num_chunks - 1ULL) / num_chunks;

        // get device pitch for workspace array (q pitch, complex Scalar)
        unsigned long long _pitch_q = get_device_pitch(chunk_size, 2 * SCALAR_SIZE);

        // add memory required for helper lags array
        mem_req += num_lags * 4ULL;

        // add memory required for workspace1 and workspace2
        mem_req += 2ULL * max(_pitch_q * length, chunk_size * pitch_t) * 16ULL;

        // add memory required for power_spec and var helper arrays
        mem_req += 2ULL * chunk_size * 16ULL;

        // check memory
        if (free_mem > mem_req)
        {
            // estimate new num_chunks
            unsigned long long new_num_chunks = (num_chunks * mem_req + free_mem - 1ULL) / free_mem;
            if (new_num_chunks == prev_num_chunks)
            {
                pitch_q = _pitch_q;
                break;
            }
            else
            {
                prev_num_chunks = num_chunks;
                num_chunks = new_num_chunks;
            }
        }
        else if (num_chunks == (nx / 2ULL + 1ULL) * ny)
        {
            throw std::runtime_error("Not enough space on GPU for structure function calculation.");
        }
        else
        {
            num_chunks = prev_num_chunks;
            unsigned long long chunk_size = ((nx / 2ULL + 1ULL) * ny + num_chunks - 1ULL) / num_chunks;
            pitch_q = get_device_pitch(chunk_size, 2 * SCALAR_SIZE);
            break;
        }
    }
}

/*!
    Optimize structure function fft execution parameters based on available gpu memory

    Writes in the corresponding arguments:
        - the number of iterations for structure function (q-vector chunks)
        - the pitch in number of elements for workspace (pitch_q, complex double)
        - the pitch in number of elements for workspace (pitch_t, complex double)
        - the pitch in number of elements for workspace (pitch_nt, complex double)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
 */
void optimize_fft(unsigned long long &pitch_q,
                  unsigned long long &pitch_t,
                  unsigned long long &pitch_nt,
                  unsigned long long &num_chunks,
                  unsigned long long length,
                  unsigned long long nx,
                  unsigned long long ny,
                  unsigned long long nt,
                  unsigned long long num_lags,
                  unsigned long long free_mem)
{
    // Calculations are done in double precision.
    // To compute the image structure function in fft mode, we need
    //  - helper lags array (unsigned int, 4 bytes)
    //      lags.size() * 4 bytes
    //  - workspace1 (complex double, 16 bytes)
    //      chunk_size * pitch_nt * 16 bytes
    //  - workspace2 (complex double, 16 bytes)
    //      max(pitch_q * length, chunk_size * pitch_t) * 16 bytes
    //  - cufft internal buffer
    //      {programmatically determined...}
    //  - prefix sum
    //      (length / 1024 + 2) * chunk_size * 16 bytes
    //  - power_spec and var arrays (double2, 16 bytes)
    //      2 * chunk_size * 16 bytes

    // memory required
    unsigned long long mem_req = 0ULL;

    // get device pitch for workspace array (time pitch, complex double)
    pitch_t = get_device_pitch(length, 16);
    // get device pitch for workspace array (fft pitch, complex double)
    pitch_nt = get_device_pitch(nt, 16);

    // start with worst case scenario:
    // we need to perform as many loop iterations as the number of q vectors
    num_chunks = (nx / 2ULL + 1ULL) * ny;
    unsigned long long prev_num_chunks;

    while (true)
    {
        // reset memory required
        mem_req = 0ULL;

        // compute the number of batched q vectors
        unsigned long long chunk_size = ((nx / 2ULL + 1ULL) * ny + num_chunks - 1ULL) / num_chunks;

        // cufft internal buffer
        cufftResult res;
        unsigned long long mem_fft = get_device_fft_mem(nt, chunk_size, pitch_nt, res);
        if (res == CUFFT_SUCCESS)
        {
            // add internal buffer memory required for cufft
            mem_req += mem_fft;

            // get device pitch for workspace array (q pitch, complex Scalar)
            unsigned long long _pitch_q = get_device_pitch(chunk_size, 2 * SCALAR_SIZE);

            // add memory required for helper lags array
            mem_req += num_lags * 4ULL;

            // add memory required for workspace1
            mem_req += chunk_size * pitch_nt * 16ULL;

            // add memory required for workspace2
            mem_req += max(_pitch_q * length, chunk_size * pitch_t) * 16ULL;

            // add memory required for power_spec and var helper arrays
            mem_req += 2ULL * chunk_size * 16ULL;

            // check memory
            if (free_mem > mem_req)
            {
                // estimate new num_fft2
                unsigned long long new_num_chunks = (num_chunks * mem_req + free_mem - 1ULL) / free_mem;
                if (new_num_chunks == prev_num_chunks)
                {
                    pitch_q = _pitch_q;
                    break;
                }
                else
                {
                    prev_num_chunks = num_chunks;
                    num_chunks = new_num_chunks;
                }
            }
            else if (num_chunks == (nx / 2ULL + 1ULL) * ny)
            {
                throw std::runtime_error("Not enough space on GPU for structure function calculation with fft.");
            }
            else
            {
                num_chunks = prev_num_chunks;
                unsigned long long chunk_size = ((nx / 2ULL + 1ULL) * ny + num_chunks - 1ULL) / num_chunks;
                pitch_q = get_device_pitch(chunk_size, 2 * SCALAR_SIZE);
                break;
            }
        }
        else if (num_chunks == (nx / 2ULL + 1ULL) * ny)
        {
            throw std::runtime_error("Not enough space on GPU for structure function calculation with fft. cufftResult #: " + res);
        }
    }
}

/*!
    Estimate device memory needed for diff mode and optimize memory usage.

    Writes in the corresponding arguments the number of iterations for:
        - fft2 (frame chunks)
        - structure function (q-vector chunks)
        - fftshift (frame chunks)
    and the pitch in number of elements for:
        - buffer array (real values)
        - workspace (pitch_q, complex double)
        - workspace (pitch_t, complex double)
        - shift workspace (pitch_fs, complex double)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
 */
void chk_device_mem_diff(unsigned long long width,
                         unsigned long long height,
                         int pixel_Nbytes,
                         unsigned long long nx,
                         unsigned long long ny,
                         unsigned long long length,
                         vector<unsigned int> lags,
                         bool is_input_double,
                         unsigned long long &num_fft2,
                         unsigned long long &num_chunks,
                         unsigned long long &num_shift,
                         unsigned long long &pitch_buff,
                         unsigned long long &pitch_nx,
                         unsigned long long &pitch_q,
                         unsigned long long &pitch_t,
                         unsigned long long &pitch_fs)
{
    // get device memory
    unsigned long long free_mem;
    get_device_free_mem(free_mem);

    // leave some free space
    free_mem = (unsigned long long)(0.9 * (double)free_mem);

    // evaluate parameters for fft2
    optimize_fft2(pitch_buff,
                  pitch_nx,
                  num_fft2,
                  is_input_double,
                  pixel_Nbytes,
                  width,
                  height,
                  length,
                  nx,
                  ny,
                  free_mem);

    // evaluate parameters for structure function diff
    optimize_diff(pitch_q,
                  pitch_t,
                  num_chunks,
                  length,
                  nx,
                  ny,
                  lags.size(),
                  free_mem);

    // evaluate parameters for shift
    optimize_shift(pitch_fs,
                   num_shift,
                   nx,
                   ny,
                   lags.size(),
                   free_mem);
}

/*!
    Estimate device memory needed for fft mode and optimize memory usage.

    Writes in the corresponding arguments the number of iterations for:
        - fft2 (frame chunks)
        - structure function (q-vector chunks)
        - fftshift (frame chunks)
    and the pitch in number of elements for:
        - buffer array (real values)
        - workspace (pitch_q, complex double)
        - workspace (pitch_t, complex double)
        - workspace (pitch_nt, complex double)
        - shift workspace (pitch_fs, complex double)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
 */
void chk_device_mem_fft(unsigned long long width,
                        unsigned long long height,
                        int pixel_Nbytes,
                        unsigned long long nx,
                        unsigned long long ny,
                        unsigned long long nt,
                        unsigned long long length,
                        vector<unsigned int> lags,
                        bool is_input_double,
                        unsigned long long &num_fft2,
                        unsigned long long &num_chunks,
                        unsigned long long &num_shift,
                        unsigned long long &pitch_buff,
                        unsigned long long &pitch_nx,
                        unsigned long long &pitch_q,
                        unsigned long long &pitch_t,
                        unsigned long long &pitch_nt,
                        unsigned long long &pitch_fs)
{
    // get device memory
    unsigned long long free_mem;
    get_device_free_mem(free_mem);

    // leave some free space
    free_mem = (unsigned long long)(0.9 * (double)free_mem);

    // evaluate parameters for fft2
    optimize_fft2(pitch_buff,
                  pitch_nx,
                  num_fft2,
                  is_input_double,
                  pixel_Nbytes,
                  width,
                  height,
                  length,
                  nx,
                  ny,
                  free_mem);

    // evaluate parameters for structure function fft
    optimize_fft(pitch_q,
                 pitch_t,
                 pitch_nt,
                 num_chunks,
                 length,
                 nx,
                 ny,
                 nt,
                 lags.size(),
                 free_mem);

    // evaluate parameters for shift
    optimize_shift(pitch_fs,
                   num_shift,
                   nx,
                   ny,
                   lags.size(),
                   free_mem);
}
