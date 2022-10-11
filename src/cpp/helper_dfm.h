// Maintainer: enrico-lattuada

// inclusion guard
#ifndef __HELPER_DFM_H__
#define __HELPER_DFM_H__

/*! \file helper_dfm.h
    \brief Declaration of helper functions for DFM calculations
*/

// *** headers ***
#include <vector>

using namespace std;

// *** code ***

/*! \brief Copy src to dest with stride
    \param src      source vector
    \param dest     destination vector
    \param start    starting index
    \param stride   stride between elements in destination
 */
void copy_vec_with_stride(vector<double> &src,
                          vector<double> &dest,
                          size_t start,
                          size_t stride);

/*! \brief Keep real part of vector
    \param vec  vector
    \param N    number of elements
 */
void complex2real(vector<double> &vec,
                  size_t N);

#endif