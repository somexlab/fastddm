// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __HELPER_DEBUG_CUH__
#define __HELPER_DEBUG_CUH__

/*! \file helper_debug.cuh
    \brief CUDA debug utilities
*/

// *** headers ***
#include <cuda_runtime.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// *** code ***

/********************/
/* CUDA ERROR CHECK */
/********************/
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            getchar();
            exit(code);
        }
    }
}

/*********************/
/* CUFFT ERROR CHECK */
/*********************/
// https://stackoverflow.com/questions/22953171/batched-ffts-using-cufftplanmany/23036876#23036876
static const char *_cudaGetErrorEnum(cufftResult error)
{
    switch (error)
    {
    case CUFFT_SUCCESS:
        return "CUFFT_SUCCESS";

    case CUFFT_INVALID_PLAN:
        return "CUFFT_INVALID_PLAN";

    case CUFFT_ALLOC_FAILED:
        return "CUFFT_ALLOC_FAILED";

    case CUFFT_INVALID_TYPE:
        return "CUFFT_INVALID_TYPE";

    case CUFFT_INVALID_VALUE:
        return "CUFFT_INVALID_VALUE";

    case CUFFT_INTERNAL_ERROR:
        return "CUFFT_INTERNAL_ERROR";

    case CUFFT_EXEC_FAILED:
        return "CUFFT_EXEC_FAILED";

    case CUFFT_SETUP_FAILED:
        return "CUFFT_SETUP_FAILED";

    case CUFFT_INVALID_SIZE:
        return "CUFFT_INVALID_SIZE";

    case CUFFT_UNALIGNED_DATA:
        return "CUFFT_UNALIGNED_DATA";
    }

    return "<unknown>";
}

#define cufftSafeCall(err) __cufftSafeCall(err, __FILE__, __LINE__)
inline void __cufftSafeCall(cufftResult err, const char *file, const int line)
{
    if (CUFFT_SUCCESS != err)
    {
        fprintf(stderr, "CUFFT error in file '%s', line %d\nerror %d: %s\nterminating!\n", __FILE__, __LINE__, err, _cudaGetErrorEnum(err));
        cudaDeviceReset();
        assert(0);
    }
}

#define cufftSilentSafeCall(err) __cufftSilentSafeCall(err, __FILE__, __LINE__)
inline void __cufftSilentSafeCall(cufftResult err, const char *file, const int line)
{
    if (CUFFT_SUCCESS != err)
    {
        assert(0);
    }
}

#endif // __HELPER_DEBUG_CUH__
