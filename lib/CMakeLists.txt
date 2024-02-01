# Build fftw3 library as a shared library
add_subdirectory(fftw-3.3.10 ${CMAKE_CURRENT_BINARY_DIR}/fftw3)

if(WIN32)
    set_target_properties(fftw3 PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY_RELEASE ${FASTDDM_OUTPUT_DIR}
    )
else()
    set_target_properties(fftw3 PROPERTIES
        PREFIX ""
        SUFFIX ".so"
        LIBRARY_OUTPUT_DIRECTORY ${FASTDDM_OUTPUT_DIR}
    )
endif(WIN32)

if(SINGLE_PRECISION)
    # Build fftw3f as a shared lib
    set(ENABLE_FLOAT ON CACHE INTERNAL "")
    add_subdirectory(fftw-3.3.10 ${CMAKE_CURRENT_BINARY_DIR}/fftw3f)

    if(WIN32)
        set_target_properties(fftw3f PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY_RELEASE ${FASTDDM_OUTPUT_DIR}
        )
    else()
        set_target_properties(fftw3f PROPERTIES
            PREFIX ""
            SUFFIX ".so"
            LIBRARY_OUTPUT_DIRECTORY ${FASTDDM_OUTPUT_DIR}
        )
    endif(WIN32)
endif(SINGLE_PRECISION)
