// SPDX-FileCopyrightText: 2023-present University of Vienna
// SPDX-FileCopyrightText: 2023-present Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino
// SPDX-License-Identifier: GPL-3.0-or-later

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

#ifndef __GPU_UTILS_CUH__
#define __GPU_UTILS_CUH__

/*! \file gpu_utils.cuh
    \brief Declaration of utility functions for GPU
*/

// *** headers ***

#include "../python_defs.h"

// *** code ***

/*! \brief Get the number of compute-capable devices
    \return Number of compute-capable devices
*/
int PYBIND11_EXPORT get_num_devices();

/*! \brief Set the device to be used
    \param device_id ID of the device to use
*/
void PYBIND11_EXPORT set_device(int device_id);

/*! \brief Get which device is currently being used
    \return ID of the currently used device
*/
int PYBIND11_EXPORT get_device();

/*! \brief Get free device memory (in bytes)
    \return Free device memory
*/
unsigned long long PYBIND11_EXPORT get_free_device_memory();

#endif // __GPU_UTILS_CUH__
