// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

#ifndef __MEMCHK_GPU_CUH__
#define __MEMCHK_GPU_CUH__

/*! \file memchk_gpu.cuh
    \brief Declaration of utility functions for GPU memory check and optimization
*/

// *** headers ***
#include "../python_defs.h"

// *** code ***

/*! \brief Set the device to be used
    \param device_id ID of the device to use
*/
void PYBIND11_EXPORT set_device(int device_id);

/*! \brief Get free device memory (in bytes)
 */
unsigned long long PYBIND11_EXPORT get_free_device_memory();

#endif // __MEMCHK_GPU_CUH__
