# translate CMake true/false to python
if(ENABLE_CPP)
    SET(IS_CPP_ENABLED True)
else()
    SET(IS_CPP_ENABLED False)
endif()

if(ENABLE_CUDA)
    SET(IS_CUDA_ENABLED True)
else()
    SET(IS_CUDA_ENABLED False)
endif()

if(ENABLE_CPP)
    SET(_fddm_sources
        cpp/ddm.cc
        cpp/helper_fftw.cc
        cpp/helper_ddm.cc
    )
endif(ENABLE_CPP)

if(ENABLE_CUDA)
    SET(_fddm_cu_sources
        cuda/ddm_cuda.cu
        cuda/helper_cufft.cu
        cuda/helper_ddm_cuda.cu
        cuda/helper_prefix_sum.cu
        cuda/helper_memchk_gpu.cu
    )

    list(APPEND _fddm_sources
         cuda/ddm_cuda.cc
         cuda/helper_memchk_gpu.cc
    )

    set_source_files_properties(${_fddm_cu_sources} PROPERTIES LANGUAGE CUDA)
endif(ENABLE_CUDA)

if(ENABLE_CPP)
    # Include external libraries
    include_directories( ${PROJECT_SOURCE_DIR}/lib/fftw-3.3.10/api )
    include_directories( ${PROJECT_SOURCE_DIR}/lib/pybind11/include )

    # Add _core library
    pybind11_add_module(_core SHARED python/bindings.cc ${_fddm_sources} ${_fddm_cu_sources})

    if(WIN32)
        target_link_libraries(_core PUBLIC ${PYTHON_LIBRARIES} fftw3)
        set_target_properties(_core PROPERTIES
                                    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${FASTDDM_OUTPUT_DIR}
                            )
    else(WIN32)
        #target_link_libraries(_core PUBLIC ${PYTHON_LIBRARIES} fftw3 m)
        target_link_libraries(_core PUBLIC fftw3 m)
        set_target_properties(_core PROPERTIES
                                    SUFFIX ".so"
                                    LIBRARY_OUTPUT_DIRECTORY ${FASTDDM_OUTPUT_DIR}
                            )
    endif(WIN32)
endif(ENABLE_CPP)

if(ENABLE_CUDA)
    include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    include(CUDAsetup.cmake)

    target_compile_definitions(_core PUBLIC ENABLE_CUDA)

    if(${CMAKE_VERSION} VERSION_LESS "3.18.0")
        target_compile_options(_core PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_${_cuda_min_arch})
    else()
        set_target_properties(_core PROPERTIES
                                    CUDA_ARCHITECTURES ${_cuda_min_arch}
                            )
    endif()

    # Add core_cuda python module
    target_link_libraries(_core PUBLIC cufft nvidia-ml)
endif(ENABLE_CUDA)

# Copy the files required to install the library with python's setuptools
configure_file(python/__init__.py
               ${FASTDDM_OUTPUT_DIR}/__init__.py)

configure_file(python/_fftopt.py
               ${FASTDDM_OUTPUT_DIR}/_fftopt.py)

if(ENABLE_CPP)
    configure_file(python/_ddm_cpp.py
                   ${FASTDDM_OUTPUT_DIR}/_ddm_cpp.py)
endif(ENABLE_CPP)

if(ENABLE_CUDA)
    configure_file(python/_ddm_cuda.py
                   ${FASTDDM_OUTPUT_DIR}/_ddm_cuda.py)

    configure_file(python/_memchk.py
                   ${FASTDDM_OUTPUT_DIR}/_memchk.py)

    configure_file(python/_gpumemchk.py
                   ${FASTDDM_OUTPUT_DIR}/_gpumemchk.py)
endif(ENABLE_CUDA)

configure_file(python/_ddm_python.py
               ${FASTDDM_OUTPUT_DIR}/_ddm_python.py)

configure_file(python/_io.py
               ${FASTDDM_OUTPUT_DIR}/_io.py)

configure_file(python/lags.py
               ${FASTDDM_OUTPUT_DIR}/lags.py)

configure_file(python/window.py
               ${FASTDDM_OUTPUT_DIR}/window.py)

configure_file(python/mask.py
               ${FASTDDM_OUTPUT_DIR}/mask.py)

configure_file(python/weights.py
               ${FASTDDM_OUTPUT_DIR}/weights.py)

configure_file(python/_utils.py
               ${FASTDDM_OUTPUT_DIR}/_utils.py)

configure_file(python/make_install_setup.py
               ${FASTDDM_OUTPUT_DIR}/setup.py)

configure_file(python/fit.py
               ${FASTDDM_OUTPUT_DIR}/fit.py)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${FASTDDM_OUTPUT_DIR})")
