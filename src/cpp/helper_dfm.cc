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
                          double *dest,
                          size_t start,
                          size_t stride)
{
    for (size_t ii = 0; ii < src.size(); ii++)
    {
        dest[ii * stride + start] = src[ii];
    }
}

/*!
    Make full image structure function from raw output
    and wap the quadrants of the isf according to fft2 convention
    (i.e., along axes x and y; leave t untouched).
    Keep only real part of vector and copy symmetric part.
    Make element contiguous in memory.
 */
void make_full_shifted_isf(double *vec,
                           size_t nx,
                           size_t ny,
                           size_t nt)
{
    // FFTshift along y only
    size_t c = ny / 2;
    size_t _nx = nx / 2 + 1;
    if (ny % 2 == 0)
    {
        for (size_t t = 0; t < nt; t++)
        {
            for (size_t x = 0; x < _nx; x++)
            {
                for (size_t y = 0; y < c; y++)
                {
                    swap(vec[2 * (t * ny * _nx + y * _nx + x)], vec[2 * (t * ny * _nx + (y + c) * _nx + x)]);
                }
            }
        }
    }
    else
    {
        double tmp = 0.0;
        for (size_t t = 0; t < nt; t++)
        {
            for (size_t x = 0; x < _nx; x++)
            {
                tmp = vec[2 * (t * ny * _nx + x)];
                for (size_t y = 0; y < c; y++)
                {
                    vec[2 * (t * ny * _nx + y * _nx + x)] = vec[2 * (t * ny * _nx + (y + c + 1) * _nx + x)];
                    vec[2 * (t * ny * _nx + (y + c + 1) * _nx + x)] = vec[2 * (t * ny * _nx + (y + 1) * _nx + x)];
                }
                vec[2 * (t * ny * _nx + c * _nx + x)] = tmp;
            }
        }
    }

    // Collapse elements to the right
    for (size_t t = 0; t < nt; t++)
    {
        for (size_t y = 0; y < ny; y++)
        {
            for (size_t x = 1; x < _nx; x++)
            {
                vec[2 * (t * ny * _nx + y * _nx + nx / 2) - x] = vec[2 * (t * ny * _nx + y * _nx + nx / 2) - 2 * x];
            }
        }
    }

    // Make full (copy missing values)
    for (size_t t = 0; t < nt; t++)
    {
        if (ny % 2 == 0)
        {
            // mirror first row
            for (size_t x = 0; x < _nx - 1; x++)
            {
                vec[2 * (t * ny * _nx) + x] = vec[2 * (t * ny * _nx + _nx - 1) - x];
            }
            // for other rows, make symmetry around center
            for (size_t y = 1; y < ny; y++)
            {
                // make symmetry around center
                for (size_t x = 0; x < _nx - 1; x++)
                {
                    vec[2 * (t * ny * _nx + y * _nx) + x] = vec[2 * (t * ny * _nx + (ny - y) * _nx) + 2 * (nx / 2) - x];
                }
            }
        }
        else
        {
            for (size_t y = 0; y < ny; y++)
            {
                // make symmetry around center
                for (size_t x = 0; x < _nx - 1; x++)
                {
                    vec[2 * (t * ny * _nx + y * _nx) + x] = vec[2 * (t * ny * _nx + (ny - y - 1) * _nx) + 2 * (nx / 2) - x];
                }
            }
        }
    }

    // Move to front
    for (size_t t = 0; t < nt; t++)
    {
        for (size_t y = 0; y < ny; y++)
        {
            for (size_t x = 0; x < nx; x++)
            {
                vec[t * ny * nx + y * nx + x] = vec[2 * (t * ny * _nx + y * _nx) + x];
            }
        }
    }
}
