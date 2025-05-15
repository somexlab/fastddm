cmake_minimum_required(VERSION 3.19)

project(fastddm)

# Set default options
option(ENABLE_CPP "Enable C++ support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(SINGLE_PRECISION "Enable single precision math" OFF)

# The default behavior is to build in Release mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Define the staging directory for the files and libraries
set(CONFIGURED_FILES_STAGING_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/python/fastddm)

# Relative path rettings
if(APPLE)
    # Enable the macOS rpath
    SET(CMAKE_MACOSX_RPATH ON)
    # Set the install rpath to the loader path
    SET(CMAKE_INSTALL_RPATH "@loader_path")
    # Build with the install rpath
    SET(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
else()
    # Use ORIGIN as the build rpath
    SET(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
endif()

# Enable C++
if(ENABLE_CPP)
    enable_language(CXX)
    # Add library subdirectory
    add_subdirectory(lib)
endif(ENABLE_CPP)

if(ENABLE_CPP OR ENABLE_CUDA)
    # Set C++ standard
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    # Find pybind11
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# Enable CUDA
if(ENABLE_CUDA)
    enable_language(CUDA)
    # Set CUDA C++ standard
    set(CMAKE_CUDA_STANDARD 11)
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    # If not set by user, let CMake try to auto-detect the architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures")
    endif()
endif(ENABLE_CUDA)

# Add source subdirectory
add_subdirectory(src)
