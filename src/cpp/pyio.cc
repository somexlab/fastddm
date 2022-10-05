// Maintainer: enrico-lattuada

/*! \file pyio.cc
    \brief Definition of python/C++ communication functions
*/

// *** headers ***
#include "pyio.h"

// *** code ***

/*!
    Copy a python numpy array to a C++ vector.

    The pixels are copied
 */
void numpy2vector(const py::buffer_info buff,
                  vector<double> &dest,
                  size_t nx,
                  size_t ny)
{
    // get pointer to values
    double *vals = (double *)buff.ptr;

    // get shape of original input
    size_t length = buff.shape[0];
    size_t height = buff.shape[1];
    size_t width = buff.shape[2];

    // copy values to vector
    for (size_t t = 0; t < length; t++)
    {
        for (size_t y = 0; y < height; y++)
        {
            /*
            for (size_t x = 0; x < width; x++)
            {
                dest[t * (nx * ny) + y * nx + x] = vals[t * (height * width) + y * width + x];
            }
             */
            // copy row
            copy(vals + t * (height * width) + y * width,
                 vals + t * (height * width) + (y + 1) * width,
                 dest.begin() + t * (nx * ny) + y * nx);
        }
    }
}