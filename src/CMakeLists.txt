SET(dfm_lib_SOURCES
    cpp/dfm.cc
    cpp/pycomm.cc
    cpp/helper_fftw.cc
    cpp/helper_dfm.cc
)

# Include external libraries
include_directories( ${PROJECT_SOURCE_DIR}/lib/fftw-3.3.10/api )
include_directories( ${PROJECT_SOURCE_DIR}/lib/pybind11/include )

# Add _dfmtoolbox library
add_library(_dfmtoolbox STATIC ${dfm_lib_SOURCES})
target_link_libraries(_dfmtoolbox ${PYTHON_LIBRARIES} fftw3 m)
set_target_properties(_dfmtoolbox PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Add core python module
pybind11_add_module(core SHARED python/bindings.cc)
target_link_libraries(core PRIVATE _dfmtoolbox)
set_target_properties(core PROPERTIES
                            SUFFIX ".so"
                            LIBRARY_OUTPUT_DIRECTORY ${DFMTOOLBOX_OUTPUT_DIR}
                    )       

# Copy the files required to install the library with python's setuptools
configure_file(python/__init__.py
               ${DFMTOOLBOX_OUTPUT_DIR}/__init__.py)

configure_file(python/_fftopt.py
               ${DFMTOOLBOX_OUTPUT_DIR}/_fftopt.py)

configure_file(python/_dfm_cpp.py
               ${DFMTOOLBOX_OUTPUT_DIR}/_dfm_cpp.py)

configure_file(python/_dfm_python.py
               ${DFMTOOLBOX_OUTPUT_DIR}/_dfm_python.py)

configure_file(python/dfm.py
               ${DFMTOOLBOX_OUTPUT_DIR}/dfm.py)

configure_file(python/imagestructurefunction.py
               ${DFMTOOLBOX_OUTPUT_DIR}/imagestructurefunction.py)

configure_file(python/lags.py
               ${DFMTOOLBOX_OUTPUT_DIR}/lags.py)

configure_file(python/window.py
               ${DFMTOOLBOX_OUTPUT_DIR}/window.py)

configure_file(python/utils.py
               ${DFMTOOLBOX_OUTPUT_DIR}/utils.py)

configure_file(python/make_install_setup.py
               ${DFMTOOLBOX_OUTPUT_DIR}/setup.py)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${DFMTOOLBOX_OUTPUT_DIR})")
