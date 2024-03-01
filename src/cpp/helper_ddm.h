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
    \param src      Source vector
    \param dest     Destination array
    \param start    Starting index
    \param stride   Stride between elements in destination
 */
void copy_vec_with_stride(vector<double> &src,
                          Scalar *dest,
                          unsigned long long start,
                          unsigned long long stride);

/*! \brief Make structure function from raw output and shift elements
    \param vec  Array
    \param nx   Number of fft nodes in x direction
    \param ny   Number of fft nodes in y direction
    \param nt   Number of frames
 */
void make_shifted_isf(Scalar *vec,
                      unsigned long long nx,
                      unsigned long long ny,
                      unsigned long long nt);

#endif // __HELPER_DDM_H__
