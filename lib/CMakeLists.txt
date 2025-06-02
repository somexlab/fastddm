# Build fftw3 library as a shared library
add_subdirectory(fftw-3.3.10 ${CMAKE_CURRENT_BINARY_DIR}/fftw3)

if(UNIX)
    set_target_properties(fftw3 PROPERTIES
        PREFIX ""
        SUFFIX ".so"
    )
endif()

# Install fftw3 library in the project directory
install(TARGETS fftw3
    LIBRARY DESTINATION fastddm
    RUNTIME DESTINATION fastddm
)

# Build fftw3f library as a shared library if SINGLE_PRECISION is enabled
if(SINGLE_PRECISION)
    # Build fftw3f as a shared lib
    set(ENABLE_FLOAT ON CACHE INTERNAL "")
    add_subdirectory(fftw-3.3.10 ${CMAKE_CURRENT_BINARY_DIR}/fftw3f)

    if(UNIX)
        set_target_properties(fftw3f PROPERTIES
            PREFIX ""
            SUFFIX ".so"
        )
    endif()

    # Install fftw3f library in the project directory
    install(TARGETS fftw3f
        LIBRARY DESTINATION fastddm
        RUNTIME DESTINATION fastddm
    )
endif(SINGLE_PRECISION)
