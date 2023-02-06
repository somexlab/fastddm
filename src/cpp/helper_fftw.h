// Maintainer: enrico-lattuada

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
    \param input    input array
    \param nx       number of fft nodes in x direction
    \param ny       number of fft nodes in y direction
    \param nt       number of elements (in t direction)
 */
FFTW_PLAN fft2_create_plan(Scalar *input,
                           size_t nx,
                           size_t ny,
                           size_t nt);

/*! \brief Create fftw plan for the complex to complex fft
    \param input    input vector
    \param nt       number of fft nodes in t direction
    \param N        number of elements
 */
FFTW_PLAN fft_create_plan(vector<Scalar> &input,
                          size_t nt,
                          size_t N);

#endif