SET(_dfmtoolbox_SOURCES
    cpp/dfm.cc
    cpp/pycomm.cc
    cpp/helper_fftw.cc
    cpp/helper_dfm.cc
    python/bindings.cc
)

include_directories( ${PROJECT_SOURCE_DIR}/lib/pybind11/include )
include_directories( ${PROJECT_SOURCE_DIR}/lib/fftw-3.3.10/api )

pybind11_add_module(core SHARED ${_dfmtoolbox_SOURCES} NO_EXTRAS)
if(NOT WIN32)
    target_link_libraries(core PRIVATE fftw3 m)
else()
    target_link_libraries(core PRIVATE fftw3)
endif()

SET(DFMTOOLBOX_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/python/dfmtoolbox")

if(NOT WIN32)
    set_target_properties(core
                            PROPERTIES
                                SUFFIX ".so"
                                LIBRARY_OUTPUT_DIRECTORY ${DFMTOOLBOX_OUTPUT_DIR}
                    )
    add_custom_command(TARGET core POST_BUILD
    				        COMMAND ${CMAKE_COMMAND} -E copy
        			        "${CMAKE_CURRENT_BINARY_DIR}/../lib/fftw-3.3.10/libfftw*"
        			        ${DFMTOOLBOX_OUTPUT_DIR}
			        )
else()
    set_target_properties(core
                            PROPERTIES
                                RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                    )
    add_custom_command(TARGET core POST_BUILD
    				        COMMAND ${CMAKE_COMMAND} -E copy_directory
        			        "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIGURATION>"
        			        ${DFMTOOLBOX_OUTPUT_DIR}
			        )
    add_custom_command(TARGET core POST_BUILD
    				        COMMAND ${CMAKE_COMMAND} -E copy
        			        "${CMAKE_CURRENT_BINARY_DIR}/../lib/fftw-3.3.10/$<CONFIGURATION>/fftw3.dll"
        			        ${DFMTOOLBOX_OUTPUT_DIR}
			        )
endif()

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

configure_file(python/make_install_setup.py
               ${DFMTOOLBOX_OUTPUT_DIR}/setup.py)

install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} -m pip install ${DFMTOOLBOX_OUTPUT_DIR})")
