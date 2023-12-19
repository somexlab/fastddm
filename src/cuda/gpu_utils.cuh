// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

#ifndef __GPU_UTILS_H__
#define __GPU_UTILS_H__

/*! \file gpu_utils.cuh
    \brief Utility functions for GPU
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

#endif // __GPU_UTILS_H__
