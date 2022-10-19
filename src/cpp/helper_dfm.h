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

/*! \brief Make full image structure function from raw output
    \param vec  vector
    \param nx   number of fft nodes in x direction
    \param ny   number of fft nodes in y direction
    \param nt   number of frames
 */
void make_full_isf(vector<double> &vec,
                   size_t nx,
                   size_t ny,
                   size_t nt);

/*! \brief Shift fft2 elements in vector (swap quadrants)
    \param vec  vector
    \param nx   number of fft nodes in x direction
    \param ny   number of fft nodes in y direction
    \param nt   number of frames
 */
void fft2_shift(vector<double> &vec,
                size_t nx,
                size_t ny,
                size_t nt);

/*! \brief Make full image structure function from raw output and shift elements
    \param vec  vector
    \param nx   number of fft nodes in x direction
    \param ny   number of fft nodes in y direction
    \param nt   number of frames
 */
void make_full_shifted_isf(vector<double> &vec,
                           size_t nx,
                           size_t ny,
                           size_t nt);

#endif