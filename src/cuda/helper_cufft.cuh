// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __HELPER_CUFFT_CUH__
#define __HELPER_CUFFT_CUH__

/*! \file helper_cufft.cuh
    \brief Declaration of cufft helper functions
*/

// *** headers ***
#include <cuda_runtime.h>
#include <cufft.h>

// *** code ***

/*! \brief Create cufft plan for the real to complex fft2
    \param nx       number of fft nodes in x direction
    \param ny       number of fft nodes in y direction
    \param batch    number of batch elements
    \param pitch    pitch of array (calculated for complex elements)
 */
cufftHandle fft2_create_plan(size_t nx,
                             size_t ny,
                             size_t batch,
                             size_t pitch);

/*! \brief Evaluate the device memory size in bytes for fft2
    \param nx           number of fft nodes in x direction
    \param ny           number of fft nodes in y direction
    \param batch        number of batch elements
    \param pitch        pitch of array (calculated for complex elements)
    \param memsize      size (in bytes) of working area for fft2
    \param cufft_res    result of cufft function
 */
void fft2_get_mem_size(size_t nx,
                       size_t ny,
                       size_t batch,
                       size_t pitch,
                       size_t *memsize,
                       cufftResult &cufft_res);

/*! \brief Create cufft plan for the complex to complex fft
    \param nt       number of fft nodes in t direction
    \param batch    number of batch elements
    \param pitch    pitch of input array
 */
cufftHandle fft_create_plan(size_t nt,
                            size_t batch,
                            size_t pitch);

/*! \brief Evaluate the device memory size in bytes for fft
    \param nt           number of fft nodes in t direction
    \param batch        number of batch elements
    \param pitch        pitch of input array
    \param memsize      size (in bytes) of working area for fft
    \param cufft_res    result of cufft function
 */
void fft_get_mem_size(size_t nt,
                      size_t batch,
                      size_t pitch,
                      size_t *memsize,
                      cufftResult &cufft_res);

#endif // __HELPER_CUFFT_CUH__
