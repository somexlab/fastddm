cmake_minimum_required(VERSION 3.18)

option(ENABLE_CPP "Set to ON to compile with C++ support" OFF)
option(ENABLE_CUDA "Set to ON to compile with CUDA support" OFF)
option(SINGLE_PRECISION "Use single precision math" OFF)

project(fastddm)

# The default behavior is to build in Release mode
set(CMAKE_BUILD_TYPE Release)

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

# The library should be in a folder with the same name to make it easier to use python distutils
set(FASTDDM_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/python/fastddm)

# Enable C++
if(ENABLE_CPP)
    enable_language(CXX)

    # Add library subdirectory
    add_subdirectory(lib)

    # Add python/pybind11 include directories
    set(PYBIND11_PYTHON_VERSION 3 CACHE STRING "Python version to use for compiling modules")
    add_subdirectory(lib/pybind11)
    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${PROJECT_SOURCE_DIR}/lib/pybind11/include)

    # Set C++ standard
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif(ENABLE_CPP)

# Enable CUDA
if(ENABLE_CUDA)
    enable_language(CUDA)

    # Set CUDA C++ standard
    set(CMAKE_CUDA_STANDARD 11)

    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
endif(ENABLE_CUDA)

# Add source subdirectory
add_subdirectory(src)
