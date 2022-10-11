// Maintainer: enrico-lattuada

/*! \file helper_fftw.h
    \brief Definition of FFTW helper functions
*/

// *** headers ***
#include "helper_fftw.h"

// *** code ***
/*
    Create the fftw plan for the fft2 of input.
 */
fftw_plan fft2_create_plan(vector<double> &input,
                           size_t nx,
                           size_t ny,
                           size_t nt)
{
    // Define parameters
    int rank = 2;                      // The rank of the fft (2 = fft2)
    int n[2] = {ny, nx};               // Dimensions
    int howmany = nt;                  // Number of fft2 to be computed
    int *inembed = NULL;               // NULL is equivalent to passing n
    int istride = 1;                   // Distance between two elements in the input
    int idist = 2 * ny * (nx / 2 + 1); // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;               // NULL is equivalent to passing n
    int ostride = 1;                   // Distance between two elements in the output
    int odist = ny * (nx / 2 + 1);     // Distance between k-th and (k+1)-th output elements
    unsigned int flags = FFTW_MEASURE; // bitwise OR ('|') of zero or more planner flags (see http://www.fftw.org/fftw3.pdf)

    // Create the fft2 plan
    fftw_plan plan = fftw_plan_many_dft_r2c(rank,
                                            n,
                                            howmany,
                                            input.data(),
                                            inembed,
                                            istride,
                                            idist,
                                            (fftw_complex *)input.data(),
                                            onembed,
                                            ostride,
                                            odist,
                                            flags);
    return plan;
}