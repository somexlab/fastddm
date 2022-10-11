// Maintainer: enrico-lattuada

/*! \file helper_dfm.cc
    \brief Definition of helper functions for DFM calculations
*/

// *** headers ***
#include "helper_dfm.h"

#include "helper_fftw.h"
#include "helper_debug.h"

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
    Keep only real part of vector and shift elements on the front side of vec
 */
void complex2real(vector<double> &vec,
                  size_t N)
{
    for (size_t n = 1; n < N; ++n)
    {
        vec[n] = vec[2*n];
    }
}