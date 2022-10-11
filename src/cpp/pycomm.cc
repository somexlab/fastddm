// Maintainer: enrico-lattuada

/*! \file pycomm.cc
    \brief Definition of python/C++ communication functions
*/

// *** headers ***
#include "pycomm.h"

// *** code ***

/*!
    Copy a python numpy array to a C++ vector.

    The parameters nx and ny allow to set the zero padding.
 */
void numpy2vector(py::buffer_info buff,
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

/*!
    Return a C++ vector as a python numpy array.
 */
py::array_t<double> vector2numpy(vector<double> *vec,
                                 size_t nx,
                                 size_t ny,
                                 size_t nt)
{
    // This snippet is to avoid copy when return vector as numpy array
    py::capsule free_when_done(vec, [](void *f)
                               {
      auto foo = reinterpret_cast<std::vector<double> *>(f);
      delete foo; });

    size_t stride_t = nx * ny * sizeof(double);
    size_t stride_y = nx * sizeof(double);
    size_t stride_x = sizeof(double);

    return py::array_t<double>({nt, ny, nx},                   // shape
                               {stride_t, stride_y, stride_x}, // C-style contiguous strides for double
                               vec->data(),                    // data pointer
                               free_when_done);
}