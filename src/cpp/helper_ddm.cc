// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

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
void copy_vec_with_stride(vector<double> &src,
                          Scalar *dest,
                          unsigned long long start,
                          unsigned long long stride)
{
    for (unsigned long long ii = 0; ii < (unsigned long long)(src.size()); ii++)
    {
        dest[ii * stride + start] = src[ii];
    }
}

/*!
    Make full structure function from raw output and swap the quadrants
    of the structure function according to fft2 convention
    (i.e., along axes x and y; leave t untouched).
    Keep only real part of vector and copy symmetric part.
    Make element contiguous in memory.
 */
void make_shifted_isf(Scalar *vec,
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
        double tmp = 0.0;
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
                vec[2 * (t * ny * _nx + y * _nx) + x] = vec[2 * (t * ny * _nx + y * _nx) + 2 * x];
            }
        }
    }

    // Move to front
    for (unsigned long long t = 0; t < nt; t++)
    {
        for (unsigned long long y = 0; y < ny; y++)
        {
            for (unsigned long long x = 0; x < _nx; x++)
            {
                vec[t * ny * _nx + y * _nx + x] = vec[2 * (t * ny * _nx + y * _nx) + x];
            }
        }
    }
}
