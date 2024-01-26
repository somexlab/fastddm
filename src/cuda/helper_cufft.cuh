// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

#ifndef __HELPER_CUFFT_CUH__
#define __HELPER_CUFFT_CUH__

/*! \file helper_cufft.cuh
    \brief Declaration of helper functions for cufft execution on GPU
*/

// *** headers ***
#include <cufft.h>

// *** code ***

/*! \brief Get the memory size needed for the work area for a 2D cufft
    \param nx           Number of grid points in x
    \param ny           Number of grid points in y
    \param batch        Number of batched transforms
    \param pitch        Pitch of device array (calculated for complex output)
    \param cufft_res    CUFFT result
    \return             Memory size required for work area
*/
unsigned long long get_fft2_device_memory_size(size_t nx,
                                               size_t ny,
                                               size_t batch,
                                               size_t pitch,
                                               cufftResult &cufft_res);

#endif // __HELPER_CUFFT_CUH__
