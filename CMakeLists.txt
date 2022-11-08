cmake_minimum_required(VERSION 3.12)

cmake_policy(SET CMP0054 NEW)

project(dfmtoolbox LANGUAGES CXX)

# The default behavior is to build with optimization
set(CMAKE_BUILD_TYPE Release)

if(APPLE)
    set(CMAKE_MACOSX_RPATH ON)
    set(CMAKE_INSTALL_RPATH "@loader_path")
    set(CMAKE_BUILD_WITH_INSTALL_RPATH ON)
else()
    set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)
endif()

# The library should be in a folder with the same name to make it easier to use python distutils
set(DFMTOOLBOX_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/python/dfmtoolbox)

# Build fftw3 as a shared lib
add_subdirectory(lib/fftw-3.3.10)
set_target_properties(fftw3 PROPERTIES
                        PREFIX ""
                        SUFFIX ".so"
                        LIBRARY_OUTPUT_DIRECTORY ${DFMTOOLBOX_OUTPUT_DIR}
                    )

# Build with python3 bindings
find_package(PythonLibs 3 REQUIRED)
add_subdirectory(lib/pybind11)
include_directories(${PYTHON_INCLUDE_DIRS})

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Add project library subdirectory
add_subdirectory(src)