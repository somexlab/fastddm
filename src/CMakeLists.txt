SET(_dfmtoolbox_SOURCES
    cpp/dfm.cc
    cpp/pycomm.cc
    cpp/helper_fftw.cc
    python/bindings.cc
)

include_directories( ${PROJECT_SOURCE_DIR}/lib/pybind11/include )
include_directories( ${FFTW_INCLUDES} )

pybind11_add_module(core SHARED ${_dfmtoolbox_SOURCES} NO_EXTRAS)
target_link_libraries(core PRIVATE fftw3 m)

SET(DFMTOOLBOX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/dfmtoolbox")

set_target_properties(core
                        PROPERTIES
                            SUFFIX ".so"
                            LIBRARY_OUTPUT_DIRECTORY ${DFMTOOLBOX_OUTPUT_DIR}
                    )

configure_file(python/__init__.py
               ${DFMTOOLBOX_OUTPUT_DIR}/__init__.py)

configure_file(python/_fftopt.py
               ${DFMTOOLBOX_OUTPUT_DIR}/_fftopt.py)

configure_file(python/dfm_cpp.py
               ${DFMTOOLBOX_OUTPUT_DIR}/dfm_cpp.py)

configure_file(python/dfm_python.py
               ${DFMTOOLBOX_OUTPUT_DIR}/dfm_python.py)

configure_file(python/dfm.py
               ${DFMTOOLBOX_OUTPUT_DIR}/dfm.py)

configure_file(python/imagestructurefunction.py
               ${DFMTOOLBOX_OUTPUT_DIR}/imagestructurefunction.py)

configure_file(python/lags.py
               ${DFMTOOLBOX_OUTPUT_DIR}/lags.py)

configure_file(python/make_install_setup.py
               ${DFMTOOLBOX_OUTPUT_DIR}/setup.py)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${DFMTOOLBOX_OUTPUT_DIR})")