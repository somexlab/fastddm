// Maintainer: enrico-lattuada

/*! \file helper_ddm.cc
    \brief Definition of helper functions for Differential Dynamic Microscopy calculations
*/

// *** headers ***
#include "helper_ddm.h"
#include "helper_fftw.h"

#include <algorithm>

// *** code ***

/*!
    Copy the elements of src to dest, so that the k-th element of src
    is at (k*stride + start)-th position in dest
 */
template <typename S>
void copy_vec_with_stride(vector<S> &src,
                          S *dest,
                          unsigned long long start,
                          unsigned long long stride)
{
    for (unsigned long long ii = 0; ii < (unsigned long long)(src.size()); ii++)
    {
        dest[ii * stride + start] = src[ii];
    }
}

template void copy_vec_with_stride<double>(vector<double> &src, double *dest, unsigned long long start, unsigned long long stride);
template void copy_vec_with_stride<float>(vector<float> &src, float *dest, unsigned long long start, unsigned long long stride);

/*!
    Make full image structure function from raw output and swap the quadrants
    of the image structure function according to fft2 convention 
    (i.e., along axes x and y; leave t untouched).
    Keep only real part of vector and copy symmetric part.
    Make element contiguous in memory.
 */
template <typename S>
void make_full_shifted_isf(S *vec,
                           unsigned long long nx,
                           unsigned long long ny,
                           unsigned long long nt)
{
    // FFTshift along y only
    unsigned long long c = ny / 2;
    unsigned long long _nx = nx / 2 + 1;
    if (ny % 2 == 0)
    {
        for (unsigned long long t = 0; t < nt; t++)
        {
            for (unsigned long long x = 0; x < _nx; x++)
            {
                for (unsigned long long y = 0; y < c; y++)
                {
                    swap(vec[2 * (t * ny * _nx + y * _nx + x)], vec[2 * (t * ny * _nx + (y + c) * _nx + x)]);
                }
            }
        }
    }
    else
    {
        S tmp = 0.0;
        for (unsigned long long t = 0; t < nt; t++)
        {
            for (unsigned long long x = 0; x < _nx; x++)
            {
                tmp = vec[2 * (t * ny * _nx + x)];
                for (unsigned long long y = 0; y < c; y++)
                {
                    vec[2 * (t * ny * _nx + y * _nx + x)] = vec[2 * (t * ny * _nx + (y + c + 1) * _nx + x)];
                    vec[2 * (t * ny * _nx + (y + c + 1) * _nx + x)] = vec[2 * (t * ny * _nx + (y + 1) * _nx + x)];
                }
                vec[2 * (t * ny * _nx + c * _nx + x)] = tmp;
            }
        }
    }

    // Collapse elements to the right
    for (unsigned long long t = 0; t < nt; t++)
    {
        for (unsigned long long y = 0; y < ny; y++)
        {
            for (unsigned long long x = 1; x < _nx; x++)
            {
                vec[2 * (t * ny * _nx + y * _nx + nx / 2) - x] = vec[2 * (t * ny * _nx + y * _nx + nx / 2) - 2 * x];
            }
        }
    }

    // Make full (copy missing values)
    for (unsigned long long t = 0; t < nt; t++)
    {
        if (ny % 2 == 0)
        {
            // mirror first row
            for (unsigned long long x = 0; x < _nx - 1; x++)
            {
                vec[2 * (t * ny * _nx) + x] = vec[2 * (t * ny * _nx + _nx - 1) - x];
            }
            // for other rows, make symmetry around center
            for (unsigned long long y = 1; y < ny; y++)
            {
                // make symmetry around center
                for (unsigned long long x = 0; x < _nx - 1; x++)
                {
                    vec[2 * (t * ny * _nx + y * _nx) + x] = vec[2 * (t * ny * _nx + (ny - y) * _nx) + 2 * (nx / 2) - x];
                }
            }
        }
        else
        {
            for (unsigned long long y = 0; y < ny; y++)
            {
                // make symmetry around center
                for (unsigned long long x = 0; x < _nx - 1; x++)
                {
                    vec[2 * (t * ny * _nx + y * _nx) + x] = vec[2 * (t * ny * _nx + (ny - y - 1) * _nx) + 2 * (nx / 2) - x];
                }
            }
        }
    }

    // Move to front
    for (unsigned long long t = 0; t < nt; t++)
    {
        for (unsigned long long y = 0; y < ny; y++)
        {
            for (unsigned long long x = 0; x < nx; x++)
            {
                vec[t * ny * nx + y * nx + x] = vec[2 * (t * ny * _nx + y * _nx) + x];
            }
        }
    }
}

template void make_full_shifted_isf<double>(double *vec, unsigned long long nx, unsigned long long ny, unsigned long long nt);
template void make_full_shifted_isf<float>(float *vec, unsigned long long nx, unsigned long long ny, unsigned long long nt);
