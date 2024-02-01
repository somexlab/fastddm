// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __HELPER_FFTW_H__
#define __HELPER_FFTW_H__

/*! \file helper_fftw.h
    \brief Declaration of FFTW helper functions
*/

// *** headers ***
#include <vector>
#include <fftw3.h>

using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
typedef fftw_plan FFTW_PLAN;
typedef fftw_complex FFTW_COMPLEX;
#else
typedef float Scalar;
typedef fftwf_plan FFTW_PLAN;
typedef fftwf_complex FFTW_COMPLEX;
#endif

// *** code ***

/*! \brief Create fftw plan for the real to complex fft2
    \param input    Input array
    \param nx       Number of fft nodes in x direction
    \param ny       Number of fft nodes in y direction
    \param nt       Number of elements (in t direction)
 */
FFTW_PLAN fft2_create_plan(Scalar *input,
                           size_t nx,
                           size_t ny,
                           size_t nt);

/*! \brief Create fftw plan for the complex to complex fft
    \param input    Input vector
    \param nt       Number of fft nodes in t direction
    \param N        Number of elements
 */
fftw_plan fft_create_plan(vector<double> &input,
                          size_t nt,
                          size_t N);

#endif // __HELPER_FFTW_H__
