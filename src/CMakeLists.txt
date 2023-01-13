if(ENABLE_CPP)
    SET(_dfm_sources
        cpp/dfm.cc
        cpp/helper_fftw.cc
        cpp/helper_dfm.cc
    )
endif(ENABLE_CPP)

if(ENABLE_CUDA)
    SET(dfm_lib_cu_SOURCES
        cuda/dfm_cuda.cu
        cuda/helper_cufft.cu
        cuda/helper_dfm_cuda.cu
        cuda/helper_prefix_sum.cu
        cuda/helper_memchk_gpu.cu
    )
endif(ENABLE_CUDA)

if(ENABLE_CPP)
    # Include external libraries
    include_directories( ${PROJECT_SOURCE_DIR}/lib/fftw-3.3.10/api )
    include_directories( ${PROJECT_SOURCE_DIR}/lib/pybind11/include )

    # Add _dfmtoolbox library
    add_library(_dfmtoolbox STATIC ${_dfm_sources})
    if(WIN32)
        target_link_libraries(_dfmtoolbox ${PYTHON_LIBRARIES} fftw3)
    else(WIN32)
        target_link_libraries(_dfmtoolbox ${PYTHON_LIBRARIES} fftw3 m)
    endif(WIN32)
    set_target_properties(_dfmtoolbox PROPERTIES POSITION_INDEPENDENT_CODE ON)

    # Add core python module
    pybind11_add_module(core SHARED python/bindings.cc)
    target_link_libraries(core PRIVATE _dfmtoolbox)
    if(WIN32)
        set_target_properties(core PROPERTIES
                                    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${DFMTOOLBOX_OUTPUT_DIR}
                            )
    else(WIN32)
        set_target_properties(core PROPERTIES
                                    SUFFIX ".so"
                                    LIBRARY_OUTPUT_DIRECTORY ${DFMTOOLBOX_OUTPUT_DIR}
                            )
    endif(WIN32)
endif(ENABLE_CPP)

if(ENABLE_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

    # Add _dfmtoolbox_cuda library
    include(select_compute_arch.cmake)
    add_library(_dfmtoolbox_cuda STATIC ${dfm_lib_cu_SOURCES})
    if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
        CUDA_SELECT_NVCC_ARCH_FLAGS(ARCH_FLAGS Auto)
        message(STATUS "Generating code for the following CUDA architectures: ${ARCH_FLAGS_readable}")
        set_target_properties(_dfmtoolbox_cuda PROPERTIES
                                    POSITION_INDEPENDENT_CODE ON
                            )
        target_compile_options(_dfmtoolbox_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${ARCH_FLAGS}>)
    else()
        CUDA_SELECT_NVCC_ARCH(ARCHS Auto)
        message(STATUS "Generating code for the following CUDA architectures: ${ARCHS_readable}")
        set_target_properties(_dfmtoolbox_cuda PROPERTIES
                                    POSITION_INDEPENDENT_CODE ON
                                    CUDA_ARCHITECTURES ${ARCHS}
                            )
    endif()

    # Add core_cuda python module
    pybind11_add_module(core_cuda SHARED cuda/dfm_cuda.cc cuda/helper_memchk_gpu.cc python/bindings_cuda.cc)
    target_link_libraries(core_cuda PRIVATE _dfmtoolbox_cuda cufft nvidia-ml)
    set_target_properties(core_cuda PROPERTIES
                                SUFFIX ".so"
                                LIBRARY_OUTPUT_DIRECTORY ${DFMTOOLBOX_OUTPUT_DIR}
                        )
endif(ENABLE_CUDA)

# Copy the files required to install the library with python's setuptools
configure_file(python/__init__.py
               ${DFMTOOLBOX_OUTPUT_DIR}/__init__.py)

configure_file(python/_fftopt.py
               ${DFMTOOLBOX_OUTPUT_DIR}/_fftopt.py)

if(ENABLE_CPP)
    configure_file(python/_dfm_cpp.py
                   ${DFMTOOLBOX_OUTPUT_DIR}/_dfm_cpp.py)
endif(ENABLE_CPP)

if(ENABLE_CUDA)
    configure_file(python/_dfm_cuda.py
                   ${DFMTOOLBOX_OUTPUT_DIR}/_dfm_cuda.py)

    configure_file(python/_memchk.py
                   ${DFMTOOLBOX_OUTPUT_DIR}/_memchk.py)

    configure_file(python/_gpumemchk.py
                   ${DFMTOOLBOX_OUTPUT_DIR}/_gpumemchk.py)
endif(ENABLE_CUDA)

configure_file(python/_dfm_python.py
               ${DFMTOOLBOX_OUTPUT_DIR}/_dfm_python.py)

configure_file(python/_io.py
               ${DFMTOOLBOX_OUTPUT_DIR}/_io.py)

configure_file(python/lags.py
               ${DFMTOOLBOX_OUTPUT_DIR}/lags.py)

configure_file(python/window.py
               ${DFMTOOLBOX_OUTPUT_DIR}/window.py)

configure_file(python/mask.py
               ${DFMTOOLBOX_OUTPUT_DIR}/mask.py)
               
configure_file(python/weights.py
               ${DFMTOOLBOX_OUTPUT_DIR}/weights.py)

configure_file(python/utils.py
               ${DFMTOOLBOX_OUTPUT_DIR}/utils.py)

configure_file(python/make_install_setup.py
               ${DFMTOOLBOX_OUTPUT_DIR}/setup.py)

configure_file(python/fit.py
               ${DFMTOOLBOX_OUTPUT_DIR}/fit.py)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${DFMTOOLBOX_OUTPUT_DIR})")
