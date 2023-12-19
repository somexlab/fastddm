// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file gpu_utils.cu
    \brief Definition of utilities for GPU
*/

// *** headers ***
#include "gpu_utils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvml.h>

// *** code ***

/*!
    Set the device to be used
*/
void PYBIND11_EXPORT set_device(int device_id)
{
    // Get number of available devices
    int deviceCount;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);

    if (status != cudaSuccess)
    {
        throw std::runtime_error("Failed to get CUDA device count. Error: " + std::string(cudaGetErrorString(status)));
    }

    // Set device
    if (device_id < 0 || device_id >= deviceCount)
    {
        throw std::runtime_error("Invalid GPU ID provided. Valid ID range: 0 <= id < " + std::to_string(deviceCount));
    }
    else
    {
        int valid_devices[] = {device_id};
        cudaSetValidDevices(valid_devices, 1);
    }
}

/*!
    Get free device memory (in bytes)
*/
unsigned long long PYBIND11_EXPORT get_free_device_memory()
{
    // Get set device
    int device_id;
    cudaGetDevice(&device_id);

    // Get device available memory
    nvmlInit_v2();
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex_v2((unsigned int)device_id, &dev) != NVML_SUCCESS)
    {
        throw std::runtime_error("Failed to get device handle. Device ID: " + std::to_string(device_id));
    }
    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(dev, &mem);
    nvmlShutdown();

    return mem.free;
}
