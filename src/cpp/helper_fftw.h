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

// *** code ***

/*! \brief Create fftw plan for the real to complex fft2
    \param input    input vector
    \param nx       number of fft nodes in x direction
    \param ny       number of fft nodes in y direction
    \param nz       number of fft nodes in z direction
 */
fftw_plan fft2_create_plan(vector<double> &input,
                           size_t nx,
                           size_t ny,
                           size_t nt);

#endif