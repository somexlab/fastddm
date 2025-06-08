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

# Set Python source files
set(PYTHON_SOURCES
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
    list(APPEND PYTHON_SOURCES
        python/_ddm_cpp.py
    )
endif(ENABLE_CPP)

# Append CUDA specific modules
if(ENABLE_CUDA)
    list(APPEND PYTHON_SOURCES
        python/_ddm_cuda.py
        python/_memchk.py
        python/_gpumemchk.py
    )
endif(ENABLE_CUDA)

# Configure Python files in the list and install them in the project directory
set(CONFIGURED_PYTHON_DIR ${PACKAGE_OUTPUT_DIR}/configured_python_files)
# Define the macro
macro(configure_and_install_py_files_to_package_root FILE_LIST_VARNAME)
    foreach(py_source_file ${${FILE_LIST_VARNAME}})
        get_filename_component(py_filename ${py_source_file} NAME)
        configure_file(${py_source_file} ${CONFIGURED_PYTHON_DIR}/${py_filename})
        install(FILES ${CONFIGURED_PYTHON_DIR}/${py_filename}
                DESTINATION fastddm)
    endforeach()
endmacro()

# Configure the Python files
configure_and_install_py_files_to_package_root(PYTHON_SOURCES)

# Manually configure and install the _config.py.in
configure_file(python/_config.py.in
    ${CONFIGURED_PYTHON_DIR}/_config.py)
install(FILES ${CONFIGURED_PYTHON_DIR}/_config.py
        DESTINATION fastddm)

# Set C++ source files
set(FASTDDM_SOURCES
    cpp/ddm.cc
    cpp/helper_fftw.cc
    cpp/helper_ddm.cc
)

# Set gpu C++ source files
set(FASTDDM_GPU_SOURCES
    cuda/memchk_gpu.cc
    cuda/ddm_cuda.cc
)

# Set gpu CUDA source files
set(FASTDDM_GPU_CU_SOURCES
    cuda/ddm_cuda.cu
    cuda/gpu_utils.cu
    cuda/memchk_gpu.cu
    cuda/helper_cufft.cu
    cuda/helper_ddm_cuda.cu
    cuda/helper_prefix_sum.cu
)

# Set language for cuda files
if(ENABLE_CUDA)
    set_source_files_properties(${FASTDDM_GPU_CU_SOURCES} PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

# Compile with C++ support
if(ENABLE_CPP)
    # Include external libraries
    include_directories(${PROJECT_SOURCE_DIR}/lib/fftw-3.3.10/api)
    # Add fddm library
    add_library(fddm SHARED ${FASTDDM_SOURCES})
    target_link_libraries(fddm PUBLIC fftw3 Python::Python)
    target_link_libraries(fddm PRIVATE pybind11::headers)
    # Link fftw3f library if SINGLE_PRECISION is enabled
    if(SINGLE_PRECISION)
        target_link_libraries(fddm PUBLIC fftw3f)
        target_compile_definitions(fddm PUBLIC SINGLE_PRECISION)
    endif(SINGLE_PRECISION)

    if(NOT WIN32)
        target_link_libraries(fddm PUBLIC m)
    endif(NOT WIN32)

    set_target_properties(fddm PROPERTIES
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

    target_link_libraries(_core PRIVATE fddm pybind11::headers)
    set_target_properties(_core PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    # Install the _core and fddm libraries in the project directory
    install(TARGETS _core fddm
        LIBRARY DESTINATION fastddm
        RUNTIME DESTINATION fastddm
    )
endif(ENABLE_CPP)

# Compile wth CUDA support
if(ENABLE_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    # Add fddm_cuda library
    add_library(fddm_cuda SHARED ${FASTDDM_GPU_SOURCES} ${FASTDDM_GPU_CU_SOURCES})
    target_link_libraries(fddm_cuda PUBLIC CUDA::cufft CUDA::nvml Python::Python)
    target_link_libraries(fddm_cuda PRIVATE pybind11::headers)
    # Note: CUDA architectures are set in the root CMakeLists, so there is no need to set them here.
    # Add the SINGLE_PRECISION flag if enabled
    if(SINGLE_PRECISION)
        target_compile_definitions(fddm_cuda PUBLIC SINGLE_PRECISION)
    endif(SINGLE_PRECISION)

    if(NOT WIN32)
        target_link_libraries(fddm_cuda PUBLIC m)
    endif(NOT WIN32)

    set_target_properties(fddm_cuda PROPERTIES
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
    target_link_libraries(_core_cuda PRIVATE fddm_cuda pybind11::headers)
    set_target_properties(_core_cuda PROPERTIES
        PREFIX "${PYTHON_MODULE_PREFIX}"
        SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
    # Install the _core_cuda and fddm_cuda libraries in the project directory
    install(TARGETS _core_cuda fddm_cuda
        LIBRARY DESTINATION fastddm
        RUNTIME DESTINATION fastddm
    )
endif(ENABLE_CUDA)
