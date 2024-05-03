# translate CMake true/false to python
if(ENABLE_CPP)
    set(IS_CPP_ENABLED True)
else()
    set(IS_CPP_ENABLED False)
endif()

if(ENABLE_CUDA)
    set(IS_CUDA_ENABLED True)
else()
    set(IS_CUDA_ENABLED False)
endif()

if(SINGLE_PRECISION)
    set(IS_SINGLE_PRECISION True)
else()
    set(IS_SINGLE_PRECISION False)
endif()

# Set fddm_SOURCES
set(fddm_SOURCES
    cpp/ddm.cc
    cpp/helper_fftw.cc
    cpp/helper_ddm.cc
)

# Set fddm_gpu_SOURCES
set(fddm_gpu_SOURCES
    cuda/memchk_gpu.cc
    cuda/ddm_cuda.cc
)

# Set fddm_gpu_cu_SOURCES
set(fddm_gpu_cu_SOURCES
    cuda/ddm_cuda.cu
    cuda/gpu_utils.cu
    cuda/memchk_gpu.cu
    cuda/helper_cufft.cu
    cuda/helper_ddm_cuda.cu
    cuda/helper_prefix_sum.cu
)

# Set language for cuda files
if(ENABLE_CUDA)
    set_source_files_properties(${fddm_gpu_cu_SOURCES} PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

# Compile with C++ support
if(ENABLE_CPP)
    # Include external libraries
    include_directories(${PROJECT_SOURCE_DIR}/lib/fftw-3.3.10/api)

    # Add fddm library
    add_library(fddm SHARED ${fddm_SOURCES})
    target_link_libraries(fddm PUBLIC fftw3)

    # Explicitly link python libraries only for Windows
    if(WIN32)
        target_link_libraries(fddm PUBLIC ${PYTHON_LIBRARIES})
    endif(WIN32)

    if(SINGLE_PRECISION)
        target_link_libraries(fddm PUBLIC fftw3f)
        target_compile_definitions(fddm PUBLIC SINGLE_PRECISION)
    endif(SINGLE_PRECISION)

    if(NOT WIN32)
        target_link_libraries(fddm PUBLIC m)
    endif(NOT WIN32)

    set_target_properties(fddm PROPERTIES

        # Linux and macOS
        LIBRARY_OUTPUT_DIRECTORY ${FASTDDM_OUTPUT_DIR}

        # Windows
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${FASTDDM_OUTPUT_DIR}
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    # On macOS, allow undefined symbols in shared libraries to be resolved dynamically at runtime.
    if(APPLE)
        set_target_properties(fddm PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
        )
    endif(APPLE)

    # Add suffix .so for UNIX systems
    if(NOT WIN32)
        set_target_properties(fddm PROPERTIES
            SUFFIX ".so"
        )
    endif(NOT WIN32)

    # Add _core library with Python bindings
    pybind11_add_module(_core MODULE python/module.cc)

    target_link_libraries(_core PRIVATE fddm)
    set_target_properties(_core PROPERTIES

        # Linux and macOS
        LIBRARY_OUTPUT_DIRECTORY ${FASTDDM_OUTPUT_DIR}

        # Windows
        # For modules, we use the library_output_directory_release
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${FASTDDM_OUTPUT_DIR}
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif(ENABLE_CPP)

# Compile wth CUDA support
if(ENABLE_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    include(CUDAsetup.cmake)

    # Add fddm_cuda library
    add_library(fddm_cuda SHARED ${fddm_gpu_SOURCES} ${fddm_gpu_cu_SOURCES})
    target_link_libraries(fddm_cuda PUBLIC CUDA::cufft CUDA::nvml)

    # Explicitly link python libraries only for Windows
    if(WIN32)
        target_link_libraries(fddm_cuda PUBLIC ${PYTHON_LIBRARIES})
    endif(WIN32)

    # Set CUDA architectures
    set_target_properties(fddm_cuda PROPERTIES
        CUDA_ARCHITECTURES ${_cuda_min_arch}
    )

    if(SINGLE_PRECISION)
        target_compile_definitions(fddm_cuda PUBLIC SINGLE_PRECISION)
    endif(SINGLE_PRECISION)

    if(NOT WIN32)
        target_link_libraries(fddm_cuda PUBLIC m)
    endif(NOT WIN32)

    set_target_properties(fddm_cuda PROPERTIES

        # Linux and macOS
        LIBRARY_OUTPUT_DIRECTORY ${FASTDDM_OUTPUT_DIR}

        # Windows
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${FASTDDM_OUTPUT_DIR}
        WINDOWS_EXPORT_ALL_SYMBOLS ON
    )

    # On macOS, allow undefined symbols in shared libraries to be resolved dynamically at runtime.
    # This should not be necessary, since CUDA is not supported on macOS, but you never know.
    if(APPLE)
        set_target_properties(fddm_cuda PROPERTIES
            LINK_FLAGS "-undefined dynamic_lookup"
        )
    endif(APPLE)

    # Add suffix .so for UNIX systems
    if(NOT WIN32)
        set_target_properties(fddm_cuda PROPERTIES
            SUFFIX ".so"
        )
    endif(NOT WIN32)

    # Add _core_cuda library with Python bindings
    pybind11_add_module(_core_cuda MODULE python/module_cuda.cc)

    target_link_libraries(_core_cuda PRIVATE fddm_cuda)
    set_target_properties(_core_cuda PROPERTIES

        # Linux and macOS
        LIBRARY_OUTPUT_DIRECTORY ${FASTDDM_OUTPUT_DIR}

        # Windows
        # For modules, we use the library_output_directory_release
        LIBRARY_OUTPUT_DIRECTORY_RELEASE ${FASTDDM_OUTPUT_DIR}
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif(ENABLE_CUDA)

# Set Python files
set(python_SOURCES
    python/__init__.py
    python/_fftopt.py
    python/imagestructurefunction.py
    python/azimuthalaverage.py
    python/intermediatescatteringfunction.py
    python/_ddm.py
    python/_ddm_python.py
    python/_io.py
    python/_io_common.py
    python/lags.py
    python/window.py
    python/mask.py
    python/weights.py
    python/utils.py
    python/fit.py
    python/fit_models.py
    python/noise_est.py
)

# Append C++ specific modules
if(ENABLE_CPP)
    list(APPEND python_SOURCES
        python/_ddm_cpp.py
    )
endif(ENABLE_CPP)

# Append CUDA specific modules
if(ENABLE_CUDA)
    list(APPEND python_SOURCES
        python/_ddm_cuda.py
        python/_memchk.py
        python/_gpumemchk.py
    )
endif(ENABLE_CUDA)

# Configure Python files in the list and write them in the output directory
# Define the macro
macro(configure_files LIST_VARNAME OUTPUT_DIR)
    foreach(file ${${LIST_VARNAME}})
        get_filename_component(filename ${file} NAME)
        configure_file(${file} ${OUTPUT_DIR}/${filename})
    endforeach()
endmacro()

# Configure the Python files
configure_files(python_SOURCES ${FASTDDM_OUTPUT_DIR})

# Manually configure the make_install_setup.py
configure_file(python/make_install_setup.py
    ${FASTDDM_OUTPUT_DIR}/setup.py)

# Manually configure the _config.py.in
configure_file(python/_config.py.in
    ${FASTDDM_OUTPUT_DIR}/_config.py)

# Install the library
install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${FASTDDM_OUTPUT_DIR})")
