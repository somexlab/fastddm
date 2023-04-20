// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __HELPER_DDM_H__
#define __HELPER_DDM_H__

/*! \file helper_ddm.h
    \brief Declaration of helper functions for Differential Dynamic Microscopy calculations
*/

// *** headers ***
#include <vector>

using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Copy src to dest with stride
    \param src      source vector
    \param dest     destination array
    \param start    starting index
    \param stride   stride between elements in destination
 */
void copy_vec_with_stride(vector<double> &src,
                          Scalar *dest,
                          unsigned long long start,
                          unsigned long long stride);

/*! \brief Make image structure function from raw output and shift elements
    \param vec  array
    \param nx   number of fft nodes in x direction
    \param ny   number of fft nodes in y direction
    \param nt   number of frames
 */
void make_shifted_isf(Scalar *vec,
                      unsigned long long nx,
                      unsigned long long ny,
                      unsigned long long nt);

#endif // __HELPER_DDM_H__
