// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file helper_fftw.h
    \brief Definition of FFTW helper functions
*/

// *** headers ***
#include "helper_fftw.h"

#ifndef SINGLE_PRECISION
fftw_plan (*Fftw_Plan_Many_Dft_r2c)(int, const int *, int, double *, const int *, int, int, fftw_complex *, const int *, int, int, unsigned int) = &fftw_plan_many_dft_r2c;
#else
fftwf_plan (*Fftw_Plan_Many_Dft_r2c)(int, const int *, int, float *, const int *, int, int, fftwf_complex *, const int *, int, int, unsigned int) = &fftwf_plan_many_dft_r2c;
#endif

// *** code ***

/*
    Create the fftw plan for the fft2 of input.
 */
FFTW_PLAN fft2_create_plan(Scalar *input,
                           size_t nx,
                           size_t ny,
                           size_t nt)
{
    // Define parameters
    int rank = 2;                             // The rank of the fft (2 = fft2)
    int n[2] = {(int)ny, (int)nx};            // Dimensions
    int howmany = (int)nt;                    // Number of fft2 to be computed
    int *inembed = NULL;                      // NULL is equivalent to passing n
    int istride = 1;                          // Distance between two elements in the input
    int idist = (int)(2 * ny * (nx / 2 + 1)); // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;                      // NULL is equivalent to passing n
    int ostride = 1;                          // Distance between two elements in the output
    int odist = (int)(ny * (nx / 2 + 1));     // Distance between k-th and (k+1)-th output elements
    unsigned int flags = FFTW_ESTIMATE;       // bitwise OR ('|') of zero or more planner flags (see http://www.fftw.org/fftw3.pdf)

    // Create the fft2 plan
    FFTW_PLAN plan = Fftw_Plan_Many_Dft_r2c(rank,
                                            n,
                                            howmany,
                                            input,
                                            inembed,
                                            istride,
                                            idist,
                                            (FFTW_COMPLEX *)input,
                                            onembed,
                                            ostride,
                                            odist,
                                            flags);
    return plan;
}

/*
    Create the fftw plan for the fft of input.
 */
fftw_plan fft_create_plan(vector<double> &input,
                          size_t nt,
                          size_t N)
{
    // Define parameters
    int rank = 1;                       // The rank of the fft (1 = fft)
    int n[1] = {(int)nt};               // Dimensions
    int howmany = (int)N;               // Number of fft2 to be computed
    int *inembed = NULL;                // NULL is equivalent to passing n
    int istride = 1;                    // Distance between two elements in the input
    int idist = (int)nt;                // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;                // NULL is equivalent to passing n
    int ostride = 1;                    // Distance between two elements in the output
    int odist = (int)nt;                // Distance between k-th and (k+1)-th output elements
    unsigned int flags = FFTW_ESTIMATE; // bitwise OR ('|') of zero or more planner flags (see http://www.fftw.org/fftw3.pdf)

    // Create the fft2 plan
    fftw_plan plan = fftw_plan_many_dft(rank,
                                        n,
                                        howmany,
                                        (fftw_complex *)input.data(),
                                        inembed,
                                        istride,
                                        idist,
                                        (fftw_complex *)input.data(),
                                        onembed,
                                        ostride,
                                        odist,
                                        FFTW_FORWARD,
                                        flags);
    return plan;
}
