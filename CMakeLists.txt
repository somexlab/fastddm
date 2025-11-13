cmake_minimum_required(VERSION 3.18)

project(fastddm)

# Set default options
option(ENABLE_CPP "Enable C++ support" OFF)
option(ENABLE_CUDA "Enable CUDA support" OFF)
option(SINGLE_PRECISION "Enable single precision math" OFF)

# If CUDA is enabled, force ENABLE_CPP to ON
if(ENABLE_CUDA)
    set(ENABLE_CPP ON CACHE BOOL "Build the core C++ library" FORCE)
endif()

# The default behavior is to build in Release mode
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Define the staging directory for the files and libraries
set(PACKAGE_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/fastddm)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PACKAGE_OUTPUT_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PACKAGE_OUTPUT_DIR})

# Relative path rettings
if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path")
else(UNIX)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
    set(CMAKE_INSTALL_RPATH "$ORIGIN")
endif()
set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)

# Enable C++
if(ENABLE_CPP)
    enable_language(CXX)

    # Add library subdirectory
    add_subdirectory(lib)
endif(ENABLE_CPP)

# Enable CUDA
if(ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 11)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA architectures
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        message(STATUS "CMAKE_CUDA_ARCHITECTURES not defined, using native architecture")
        set(CMAKE_CUDA_ARCHITECTURES native)
    else()
        message(STATUS "Using user-defined CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif(ENABLE_CUDA)

if(ENABLE_CPP OR ENABLE_CUDA)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # Find pybind11
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 CONFIG REQUIRED)
    find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
endif()

# Add source subdirectory
add_subdirectory(src)
