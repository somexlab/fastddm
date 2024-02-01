# Find CUDA libraries and binaries

set(REQUIRED_CUDA_LIB_VARS "")

# find CUDA library path
get_filename_component(CUDA_BIN_PATH ${CMAKE_CUDA_COMPILER} DIRECTORY)
get_filename_component(CUDA_LIB_PATH "${CUDA_BIN_PATH}/../lib64/" ABSOLUTE)

# Find cudart
find_library(CUDA_cudart_LIBRARY cudart HINTS ${CUDA_LIB_PATH})
mark_as_advanced(CUDA_cudart_LIBRARY)

if(CUDA_cudart_LIBRARY AND NOT TARGET CUDA::cudart)
    add_library(CUDA::cudart UNKNOWN IMPORTED)
    set_target_properties(CUDA::cudart PROPERTIES
        IMPORTED_LOCATION "${CUDA_cudart_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )
endif()

find_package_message(CUDALibsCUDART "Found cudart: ${CUDA_cudart_LIBRARY}" "[${CUDA_cudart_LIBRARY}]")
list(APPEND REQUIRED_CUDA_LIB_VARS "CUDA_cudart_LIBRARY")

# Find cufft
find_library(CUDA_cufft_LIBRARY cufft HINTS ${CUDA_LIB_PATH})
mark_as_advanced(CUDA_cufft_LIBRARY)

if(CUDA_cufft_LIBRARY AND NOT TARGET CUDA::cufft)
    add_library(CUDA::cufft UNKNOWN IMPORTED)
    set_target_properties(CUDA::cufft PROPERTIES
        IMPORTED_LOCATION "${CUDA_cufft_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )
endif()

find_package_message(CUDALibsCUFFT "Found cufft: ${CUDA_cufft_LIBRARY}" "[${CUDA_cufft_LIBRARY}]")
list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_cufft_LIBRARY)

# Find nvidia-ml
find_library(CUDA_nvml_LIBRARY NAMES nvidia-ml nvml HINTS ${CUDA_LIB_PATH})
mark_as_advanced(CUDA_nvml_LIBRARY)

if(CUDA_nvml_LIBRARY AND NOT TARGET CUDA::nvml)
    add_library(CUDA::nvml UNKNOWN IMPORTED)
    set_target_properties(CUDA::nvml PROPERTIES
        IMPORTED_LOCATION "${CUDA_nvml_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    )
endif()

find_package_message(CUDALibsNVML "Found nvml: ${CUDA_nvml_LIBRARY}" "[${CUDA_nvml_LIBRARY}]")
list(APPEND REQUIRED_CUDA_LIB_VARS CUDA_nvml_LIBRARY)
