// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file gpu_utils.cu
    \brief Definition of utility functions for GPU
*/

// *** headers ***
#include "gpu_utils.cuh"

#include <cuda_runtime.h>
#include <nvml.h>

// *** code ***

/*!
    Get the number of compute-capable devices
*/
int PYBIND11_EXPORT get_num_devices()
{
    // Get number of available devices
    int deviceCount;
    cudaError_t status = cudaGetDeviceCount(&deviceCount);

    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            "Failed to get CUDA device count. Error: " + std::string(cudaGetErrorString(status)));
    }

    return deviceCount;
}

/*!
    Set the device to be used
*/
void PYBIND11_EXPORT set_device(int device_id)
{
    // Get number of available devices
    int deviceCount = get_num_devices();

    if (device_id < 0 || device_id >= deviceCount)
    {
        throw std::runtime_error(
            "Invalid GPU ID provided. Valid ID range: 0 <= id < " + std::to_string(deviceCount));
    }

    // Set device
    int valid_devices[] = {device_id};
    cudaError_t status = cudaSetValidDevices(valid_devices, 1);

    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            "Failed to set CUDA device. Error: " + std::string(cudaGetErrorString(status)));
    }
}

/*!
    Get which device is currently being used
*/
int PYBIND11_EXPORT get_device()
{
    // Get device currently being used
    int device_id;
    cudaError_t status = cudaGetDevice(&device_id);

    if (status != cudaSuccess)
    {
        throw std::runtime_error(
            "Failed to get CUDA device. Error: " + std::string(cudaGetErrorString(status)));
    }

    return device_id;
}

/*!
    Get free device memory (in bytes)
*/
unsigned long long PYBIND11_EXPORT get_free_device_memory()
{
    // Get device currently being used
    int device_id = get_device();

    // Initialize nvml
    nvmlReturn_t status = nvmlInit_v2();

    if (status != NVML_SUCCESS)
    {
        throw std::runtime_error(
            "Failed to initialize NVML. Error: " + std::string(nvmlErrorString(status)));
    }

    // Get free memory
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex_v2((unsigned int)device_id, &dev) != NVML_SUCCESS)
    {
        throw std::runtime_error(
            "Failed to get device handle. Device ID: " + std::to_string(device_id));
    }
    nvmlMemory_t mem;
    status = nvmlDeviceGetMemoryInfo(dev, &mem);

    if (status != NVML_SUCCESS)
    {
        throw std::runtime_error(
            "Failed to get device memory info. Error: " + std::string(nvmlErrorString(status)));
    }

    nvmlShutdown();

    return mem.free;
}
