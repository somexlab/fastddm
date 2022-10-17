// Maintainer: enrico-lattuada

/*! \file helper_dfm.cc
    \brief Definition of helper functions for DFM calculations
*/

// *** headers ***
#include "helper_dfm.h"
#include "helper_fftw.h"

#include <algorithm>

// *** code ***

/*!
    Copy the elements of src to dest, so that the k-th element of src
    is at (k*stride + start)-th position in dest
 */
void copy_vec_with_stride(vector<double> &src,
                          vector<double> &dest,
                          size_t start,
                          size_t stride)
{
    for (size_t ii = 0; ii < src.size(); ii++)
    {
        dest[ii * stride + start] = src[ii];
    }
}

/*!
    Make full image structure function from raw output.
    Keep only real part of vector and copy symmetric part.
    Make element contiguous in memory.
 */
void make_full_isf(vector<double> &vec,
                   size_t nx,
                   size_t ny,
                   size_t nt)
{
    for (size_t t = 0; t < nt; t++)
    {
        // move elements to the front
        for (size_t y = 0; y < ny; y++)
        {
            for (size_t x = 0; x < (nx / 2 + 1); x++)
            {
                vec[t * ny * nx + y * nx + x] = vec[t * ny * 2 * (nx / 2 + 1) + y * 2 * (nx / 2 + 1) + 2 * x];
            }
        }

        // copy symmetric part
        for (size_t y = 0; y < ny; ++y)
        {
            size_t yy = (ny - y) % ny;

            for (size_t x = 1; x < (nx + 1) / 2; ++x)
            {
                vec[t * ny * nx + (yy + 1) * nx - x] = vec[t * ny * nx + y * nx + x];
            }
        }
    }
}

/*!
    Swap the quadrants of the isf according to fft2 convention
    (i.e., along axes x and y; leave t untouched)
 */
void fft2_shift(vector<double> &vec,
                size_t nx,
                size_t ny,
                size_t nt)
{
    // fftshift array over 2nd and 3rd axes
    // shift over x
    size_t c = nx / 2;
    if (nx % 2 == 0)
    {
        for (size_t t = 0; t < nt; t++)
        {
            for (size_t y = 0; y < ny; y++)
            {
                for (size_t x = 0; x < c; x++)
                {
                    swap(vec[t * ny * nx + y * nx + x], vec[t * ny * nx + y * nx + x + c]);
                }
            }
        }
    }
    else
    {
        double tmp = 0.0;
        for (size_t t = 0; t < nt; t++)
        {
            for (size_t y = 0; y < ny; y++)
            {
                tmp = vec[t * ny * nx + y * nx];
                for (size_t x = 0; x < c; x++)
                {
                    vec[t * ny * nx + y * nx + x] = vec[t * ny * nx + y * nx + x + c + 1];
                    vec[t * ny * nx + y * nx + x + c + 1] = vec[t * ny * nx + y * nx + x + 1];
                }
                vec[t * ny * nx + y * nx + c] = tmp;
            }
        }
    }

    // shift over y
    c = ny / 2;
    if (ny % 2 == 0)
    {
        for (size_t t = 0; t < nt; t++)
        {
            for (size_t x = 0; x < nx; x++)
            {
                for (size_t y = 0; y < c; y++)
                {
                    swap(vec[t * ny * nx + y * nx + x], vec[t * ny * nx + (y + c) * nx + x]);
                }
            }
        }
    }
    else
    {
        double tmp = 0.0;
        for (size_t t = 0; t < nt; t++)
        {
            for (size_t x = 0; x < nx; x++)
            {
                tmp = vec[t * ny * nx + x];
                for (size_t y = 0; y < c; y++)
                {
                    vec[t * ny * nx + y * nx + x] = vec[t * ny * nx + (y + c + 1) * nx + x];
                    vec[t * ny * nx + (y + c + 1) * nx + x] = vec[t * ny * nx + (y + 1) * nx + x];
                }
                vec[t * ny * nx + c * nx + x] = tmp;
            }
        }
    }
    /*
    // fftshift array over 2nd and 3rd axes
    for (size_t t = 0; t < nt; ++t)
    {
        // shift over x
        size_t c = nx / 2;
        for (size_t y = 0; y < ny; ++y)
        {
            if (nx % 2 == 0)
            {
                for (size_t x = 0; x < c; ++x)
                {
                    swap(vec[t * ny * nx + y * nx + x], vec[t * ny * nx + y * nx + x + c]);
                }
            }
            else
            {
                double tmp = vec[t * ny * nx + y * nx];
                for (size_t x = 0; x < c; ++x)
                {
                    vec[t * ny * nx + y * nx + x] = vec[t * ny * nx + y * nx + x + c + 1];
                    vec[t * ny * nx + y * nx + x + c + 1] = vec[t * ny * nx + y * nx + x + 1];
                }
                vec[t * ny * nx + y * nx + c] = tmp;
            }
        }

        // shift over y
        c = ny / 2;
        for (size_t x = 0; x < nx; ++x)
        {
            if (ny % 2 == 0)
            {
                for (size_t y = 0; y < c; ++y)
                {
                    swap(vec[t * ny * nx + y * nx + x], vec[t * ny * nx + (y + c) * nx + x]);
                }
            }
            else
            {
                double tmp = vec[t * ny * nx + x];
                for (size_t y = 0; y < c; ++y)
                {
                    vec[t * ny * nx + y * nx + x] = vec[t * ny * nx + (y + c + 1) * nx + x];
                    vec[t * ny * nx + (y + c + 1) * nx + x] = vec[t * ny * nx + (y + 1) * nx + x];
                }
                vec[t * ny * nx + c * nx + x] = tmp;
            }
        }
    }
     */
}