// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __HELPER_DDM_H__
#define __HELPER_DDM_H__

/*! \file helper_ddm.h
    \brief Declaration of helper functions for Differential Dynamic Microscopy calculations
*/

// *** headers ***
#include <vector>

using namespace std;

// *** code ***

/*! \brief Copy src to dest with stride
    \param src      source vector
    \param dest     destination array
    \param start    starting index
    \param stride   stride between elements in destination
 */
template <typename S>
void copy_vec_with_stride(vector<double> &src,
                          S *dest,
                          unsigned long long start,
                          unsigned long long stride);

/*! \brief Make full image structure function from raw output and shift elements
    \param vec  array
    \param nx   number of fft nodes in x direction
    \param ny   number of fft nodes in y direction
    \param nt   number of frames
 */
template <typename S>
void make_full_shifted_isf(S *vec,
                           unsigned long long nx,
                           unsigned long long ny,
                           unsigned long long nt);

#endif