// Maintainer: enrico-lattuada

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
#else
#include <sys/sysinfo.h>
#endif

#include <stdexcept>

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
#else
    struct sysinfo info;
    sysinfo(&info);
    free_mem = info.freeram;
#endif
}

/*!
    Estimate and check host memory needed for direct mode
 */
void chk_host_mem_direct(unsigned long long nx,
                         unsigned long long ny,
                         unsigned long long length,
                         vector<unsigned int> lags)
{
    unsigned long long free_mem;
    get_host_free_mem(free_mem);

    // leave some free space
    free_mem = (unsigned long long)(0.95 * (double)free_mem);

    /*
    Calculations are done in double precision.
    - The store the output, we need
        nx * ny * #lags * 8 bytes

    - To store the fft2, we need
        (nx / 2 + 1) * ny * length * 16 bytes
      This is always larger (or equal) than the output size.
      We then use this space as a workspace for both the fft2
      intermediate output and the final result (output is then cropped).
     */
    unsigned long long mem_required = 0;

    mem_required += (nx / 2ULL + 1ULL) * ny * length * 16ULL;

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
                      unsigned long long length)
{
    unsigned long long free_mem;
    get_host_free_mem(free_mem);

    // leave some free space
    free_mem = (unsigned long long)(0.95 * (double)free_mem);

    /*
    Calculations are done in double precision.
    - The store the output, we need
        nx * ny * #lags * 8 bytes

    - To store the fft2, we need
        (nx / 2 + 1) * ny * length * 16 bytes
      This is always larger (or equal) than the output size.
      We then use this space as a workspace for both the fft2
      intermediate output and the final result (output is then cropped).
     */
    unsigned long long mem_required = 0;

    mem_required += (nx / 2ULL + 1ULL) * ny * length * 16ULL;

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
                                       cufftResult &cufft_res)
{
    // The following line should be changed to move to workstations with multiple GPUs
    size_t memsize[1]; // We are only considering workstations with 1 GPU
    cudaGetFft2MemSize(nx,
                       ny,
                       batch,
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
    Estimate device memory needed for direct mode and optimize memory usage.

    Writes in the corresponding arguments the number of iterations for:
        - fft2 (frame chunks)
        - correlation (q-vector chunks)
        - full and fftshift (frame chunks)
    and the pitch in number of elements for:
        - buffer array (real values)
        - workspace (pitch_q, complex double)
        - workspace (pitch_t, complex double)
        - full&shift workspace (pitch_fs, complex double)

    Throws a runtime_error if the memory is not sufficient
    to perform the calculations.
 */
void chk_device_mem_direct(unsigned long long width,
                           unsigned long long height,
                           int pixel_Nbytes,
                           unsigned long long nx,
                           unsigned long long ny,
                           unsigned long long length,
                           vector<unsigned int> lags,
                           bool is_input_double,
                           unsigned long long &num_fft2,
                           unsigned long long &num_chunks,
                           unsigned long long &num_fullshift,
                           unsigned long long &pitch_buff,
                           unsigned long long &pitch_q,
                           unsigned long long &pitch_t,
                           unsigned long long &pitch_fs)
{
    // get device memory
    unsigned long long free_mem;
    get_device_free_mem(free_mem);

    // leave some free space
    free_mem = (unsigned long long)(0.9 * (double)free_mem);

    // Calculations are done in double precision.
    // To compute the fft2, we need
    //  - for the buffer (only if input is not double):
    //      pitch_x * height * fft2_batch_len * pixel_Nbytes bytes
    //  - for the workspace (complex double, 16 bytes):
    //      (nx / 2 + 1) * ny * fft2_batch_len * 16 bytes
    //  - for the cufft2 internal buffer:
    //      {programmatically determined...}
    unsigned long long mem_req = 0ULL;

    // get device pitch for buffer array (only if input is not double)
    unsigned long long _pitch_buff = 0ULL;
    if (!is_input_double)
    {
        _pitch_buff = get_device_pitch(width, pixel_Nbytes);
    }

    for (unsigned long long num = 1ULL; num < (length + 1ULL) / 2ULL + 1ULL; num++)
    {
        mem_req = 0ULL;
        // compute the number of batched fft2
        unsigned long long fft2_batch_len = (length + num - 1ULL) / num;

        // cufft2 internal buffer
        cufftResult res;
        unsigned long long mem_fft2 = get_device_fft2_mem(nx, ny, fft2_batch_len, res);
        if (res == CUFFT_SUCCESS)
        {
            mem_req += mem_fft2;

            // buffer
            if (!is_input_double)
            {
                mem_req += _pitch_buff * height * fft2_batch_len * (unsigned long long)pixel_Nbytes;
            }

            // workspace
            mem_req += (nx / 2ULL + 1ULL) * ny * fft2_batch_len * 16ULL;

            // check memory
            if (free_mem > mem_req)
            {
                num_fft2 = num;
                pitch_buff = _pitch_buff;
                break;
            }
        }
    }
    // do last check (worst case scenario, 1 fft2 per frame [i.e., fft2_batch_len = 1])
    if (free_mem <= mem_req)
    {
        mem_req = 0ULL;

        // cufft2 internal buffer
        cufftResult res;
        unsigned long long mem_fft2 = get_device_fft2_mem(nx, ny, 1, res);

        if (res == CUFFT_SUCCESS)
        {
            mem_req += mem_fft2;

            // buffer
            if (!is_input_double)
            {
                mem_req += _pitch_buff * height * (unsigned long long)pixel_Nbytes;
            }

            // workspace
            mem_req += (nx / 2ULL + 1ULL) * ny * 16ULL;

            if (free_mem > mem_req)
            {
                num_fft2 = length;
                pitch_buff = _pitch_buff;
            }
            else
            {
                throw std::runtime_error("Not enough space on GPU for fft2.");
            }
        }
        else
        {
            throw std::runtime_error("Not enough space on GPU for fft2. cufftResult #: " + res);
        }
    }

    // To compute the image structure function in direct mode, we need
    //  - helper lags array (unsigned int, 4 bytes)
    //      lags.size() * 4 bytes
    //  - workspace1 and workspace2 (complex double, 16 bytes)
    //      max(pitch_q * length, chunk_size * pitch_t) * 16 bytes

    // get device pitch for workspace array (time pitch, complex double)
    unsigned long long _pitch_t = get_device_pitch(length, 16);
    unsigned long long _pitch_q;

    for (unsigned long long num = 1ULL; num < ((nx / 2ULL + 1ULL) * ny + 1ULL) / 2ULL + 1ULL; num++)
    {
        mem_req = 0ULL;

        // compute the number of batched q vectors
        unsigned long long chunk_size = ((nx / 2ULL + 1ULL) * ny + num - 1ULL) / num;
        // get device pitch for workspace array (q pitch, complex double)
        _pitch_q = get_device_pitch(chunk_size, 16);

        // lags
        mem_req += (unsigned long long)lags.size() * 4ULL;

        // workspace1 and workspace2
        mem_req += 2ULL * max(_pitch_q * length, chunk_size * _pitch_t) * 16ULL;

        // check memory
        if (free_mem > mem_req)
        {
            num_chunks = num;
            pitch_t = _pitch_t;
            pitch_q = _pitch_q;
            break;
        }
    }
    // do last check (worst case scenario, 1 q vector per loop iteration [i.e., chunk_size = 1])
    if (free_mem <= mem_req)
    {
        mem_req = 0ULL;

        // get device pitch for workspace array (q pitch, complex double)
        _pitch_q = get_device_pitch(1, 16);

        // lags
        mem_req += (unsigned long long)lags.size() * 4ULL;

        // workspace1 and workspace2
        mem_req += 2ULL * max(_pitch_q * length, _pitch_t) * 16ULL;

        if (free_mem > mem_req)
        {
            num_chunks = (nx / 2ULL + 1ULL) * ny;
            pitch_t = _pitch_t;
            pitch_q = _pitch_q;
        }
        else
        {
            throw std::runtime_error("Not enough space on GPU for correlation.");
        }
    }

    // To compute the full and shift conversion, we need
    //  - workspace1 and workspace2 (complex double, 16 bytes)
    //      pitch_fs * ny * fullshift_batch_len * 16 bytes

    // get device pitch for workspace array (full&shift pitch, complex double)
    unsigned long long _pitch_fs = get_device_pitch((nx / 2ULL + 1ULL), 16);

    for (unsigned long long num = 1ULL; num < ((unsigned long long)lags.size() + 1ULL) / 2ULL + 1ULL; num++)
    {
        mem_req = 0ULL;

        // compute the number of batched q vectors
        unsigned long long fullshift_batch_len = ((unsigned long long)lags.size() + num - 1ULL) / num;

        // workspace1 and workspace2
        mem_req += 2ULL * _pitch_fs * ny * fullshift_batch_len * 16ULL;

        // check memory
        if (free_mem > mem_req)
        {
            num_fullshift = num;
            pitch_fs = _pitch_fs;
            break;
        }
    }
    // do last check (worst case scenario, 1 full&shift per lag [i.e., fullshift_batch_len = 1])
    if (free_mem <= mem_req)
    {
        mem_req = 0ULL;

        // workspace1 and workspace2
        mem_req += 2ULL * _pitch_fs * ny * 16ULL;

        if (free_mem > mem_req)
        {
            num_fullshift = lags.size();
            pitch_fs = _pitch_fs;
        }
        else
        {
            throw std::runtime_error("Not enough space on GPU for full and shifted power spectrum.");
        }
    }
}

/*!
    Estimate device memory needed for fft mode and optimize memory usage.

    Writes in the corresponding arguments the number of iterations for:
        - fft2 (frame chunks)
        - correlation (q-vector chunks)
        - full and fftshift (frame chunks)
    and the pitch in number of elements for:
        - buffer array (real values)
        - workspace (pitch_q, complex double)
        - workspace (pitch_t, complex double)
        - workspace (pitch_nt, complex double)
        - full&shift workspace (pitch_fs, complex double)

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
                        unsigned long long &num_fullshift,
                        unsigned long long &pitch_buff,
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

    // Calculations are done in double precision.
    // To compute the fft2, we need
    //  - for the buffer (only if input is not double):
    //      pitch_x * height * fft2_batch_len * pixel_Nbytes bytes
    //  - for the workspace (complex double, 16 bytes):
    //      (nx / 2 + 1) * ny * fft2_batch_len * 16 bytes
    //  - for the cufft2 internal buffer:
    //      {programmatically determined...}
    unsigned long long mem_req = 0ULL;

    // get device pitch for buffer array (only if input is not double)
    unsigned long long _pitch_buff = 0ULL;
    if (!is_input_double)
    {
        _pitch_buff = get_device_pitch(width, pixel_Nbytes);
    }

    for (unsigned long long num = 1ULL; num < (length + 1ULL) / 2ULL + 1ULL; num++)
    {
        mem_req = 0ULL;
        // compute the number of batched fft2
        unsigned long long fft2_batch_len = (length + num - 1ULL) / num;

        // cufft2 internal buffer
        cufftResult res;
        unsigned long long mem_fft2 = get_device_fft2_mem(nx, ny, fft2_batch_len, res);
        if (res == CUFFT_SUCCESS)
        {
            mem_req += mem_fft2;

            // buffer
            if (!is_input_double)
            {
                mem_req += _pitch_buff * height * fft2_batch_len * (unsigned long long)pixel_Nbytes;
            }

            // workspace
            mem_req += (nx / 2ULL + 1ULL) * ny * fft2_batch_len * 16ULL;

            // check memory
            if (free_mem > mem_req)
            {
                num_fft2 = num;
                pitch_buff = _pitch_buff;
                break;
            }
        }
    }
    // do last check (worst case scenario, 1 fft2 per frame [i.e., fft2_batch_len = 1])
    if (free_mem <= mem_req)
    {
        mem_req = 0ULL;

        // cufft2 internal buffer
        cufftResult res;
        unsigned long long mem_fft2 = get_device_fft2_mem(nx, ny, 1, res);

        if (res == CUFFT_SUCCESS)
        {
            mem_req += mem_fft2;

            // buffer
            if (!is_input_double)
            {
                mem_req += _pitch_buff * height * (unsigned long long)pixel_Nbytes;
            }

            // workspace
            mem_req += (nx / 2ULL + 1ULL) * ny * 16ULL;

            if (free_mem > mem_req)
            {
                num_fft2 = length;
                pitch_buff = _pitch_buff;
            }
            else
            {
                throw std::runtime_error("Not enough space on GPU for fft2.");
            }
        }
        else
        {
            throw std::runtime_error("Not enough space on GPU for fft2. cufftResult #: " + res);
        }
    }

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

    // get device pitch for workspace array (time pitch, complex double)
    unsigned long long _pitch_t = get_device_pitch(length, 16);
    // get device pitch for workspace array (fft pitch, complex double)
    unsigned long long _pitch_nt = get_device_pitch(nt, 16);
    unsigned long long _pitch_q;

    for (unsigned long long num = 1ULL; num < ((nx / 2ULL + 1ULL) * ny + 1ULL) / 2ULL + 1ULL; num++)
    {
        mem_req = 0ULL;

        // compute the number of batched q vectors
        unsigned long long chunk_size = ((nx / 2ULL + 1ULL) * ny + num - 1ULL) / num;

        // cufft internal buffer
        cufftResult res;
        unsigned long long mem_fft = get_device_fft_mem(nt, chunk_size, _pitch_nt, res);

        if (res == CUFFT_SUCCESS)
        {
            mem_req += mem_fft;

            // get device pitch for workspace array (q pitch, complex double)
            _pitch_q = get_device_pitch(chunk_size, 16);

            // lags
            mem_req += (unsigned long long)lags.size() * 4ULL;

            // workspace1
            mem_req += chunk_size * _pitch_nt * 16ULL;

            // workspace2
            mem_req += max(_pitch_q * length, chunk_size * _pitch_t) * 16ULL;

            // check memory
            if (free_mem > mem_req)
            {
                num_chunks = num;
                pitch_q = _pitch_q;
                pitch_t = _pitch_t;
                pitch_nt = _pitch_nt;
                break;
            }
        }
    }
    // do last check (worst case scenario, 1 q vector per loop iteration [i.e., chunk_size = 1])
    if (free_mem <= mem_req)
    {
        mem_req = 0ULL;

        // cufft internal buffer
        cufftResult res;
        unsigned long long mem_fft = get_device_fft_mem(nt, 1, _pitch_nt, res);

        if (res == CUFFT_SUCCESS)
        {
            mem_req += mem_fft;

            // get device pitch for workspace array (q pitch, complex double)
            _pitch_q = get_device_pitch(1, 16);

            // lags
            mem_req += lags.size() * 4;

            // workspace1
            mem_req += _pitch_nt * 16ULL;

            // workspace2
            mem_req += max(_pitch_q * length, _pitch_t) * 16ULL;

            if (free_mem > mem_req)
            {
                num_chunks = (nx / 2ULL + 1ULL) * ny;
                pitch_t = _pitch_t;
                pitch_q = _pitch_q;
                pitch_nt = _pitch_nt;
            }
            else
            {
                throw std::runtime_error("Not enough space on GPU for correlation with fft.");
            }
        }
        else
        {
            throw std::runtime_error("Not enough space on GPU for correlation with fft. cufftResult #: " + res);
        }
    }

    // To compute the full and shift conversion, we need
    //  - workspace1 and workspace2 (complex double, 16 bytes)
    //      pitch_fs * ny * fullshift_batch_len * 16 bytes

    // get device pitch for workspace array (full&shift pitch, complex double)
    unsigned long long _pitch_fs = get_device_pitch((nx / 2ULL + 1ULL), 16);

    for (unsigned long long num = 1ULL; num < ((unsigned long long)lags.size() + 1ULL) / 2ULL + 1ULL; num++)
    {
        mem_req = 0ULL;

        // compute the number of batched q vectors
        unsigned long long fullshift_batch_len = ((unsigned long long)lags.size() + num - 1ULL) / num;

        // workspace1 and workspace2
        mem_req += 2ULL * _pitch_fs * ny * fullshift_batch_len * 16ULL;

        // check memory
        if (free_mem > mem_req)
        {
            num_fullshift = num;
            pitch_fs = _pitch_fs;
            break;
        }
    }
    // do last check (worst case scenario, 1 full&shift per lag [i.e., fullshift_batch_len = 1])
    if (free_mem <= mem_req)
    {
        mem_req = 0ULL;

        // workspace1 and workspace2
        mem_req += 2ULL * _pitch_fs * ny * 16ULL;

        if (free_mem > mem_req)
        {
            num_fullshift = lags.size();
            pitch_fs = _pitch_fs;
        }
        else
        {
            throw std::runtime_error("Not enough space on GPU for full and shifted power spectrum.");
        }
    }
}