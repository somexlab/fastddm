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
    int rank = 2;                             // The rank of the fft (2 = fft2)
    int n[2] = {(int)ny, (int)nx};            // Dimensions
    int howmany = (int)nt;                    // Number of fft2 to be computed
    int *inembed = NULL;                      // NULL is equivalent to passing n
    int istride = 1;                          // Distance between two elements in the input
    int idist = (int)(2 * ny * (nx / 2 + 1)); // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;                      // NULL is equivalent to passing n
    int ostride = 1;                          // Distance between two elements in the output
    int odist = (int)(ny * (nx / 2 + 1));     // Distance between k-th and (k+1)-th output elements
    unsigned int flags = FFTW_MEASURE;        // bitwise OR ('|') of zero or more planner flags (see http://www.fftw.org/fftw3.pdf)

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

/*
    Create the fftw plan for the fft of input.
 */
fftw_plan fft_create_plan(vector<double> &input,
                          size_t nt,
                          size_t N)
{
    // Define parameters
    int rank = 1;                      // The rank of the fft (1 = fft)
    int n[1] = {(int)nt};              // Dimensions
    int howmany = (int)N;              // Number of fft2 to be computed
    int *inembed = NULL;               // NULL is equivalent to passing n
    int istride = 1;                   // Distance between two elements in the input
    int idist = (int)nt;               // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;               // NULL is equivalent to passing n
    int ostride = 1;                   // Distance between two elements in the output
    int odist = (int)nt;               // Distance between k-th and (k+1)-th output elements
    unsigned int flags = FFTW_MEASURE; // bitwise OR ('|') of zero or more planner flags (see http://www.fftw.org/fftw3.pdf)

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

/*
    Create the fftw plan for the ifft of input.
 */
fftw_plan ifft_create_plan(vector<double> &input,
                           size_t nt,
                           size_t N)
{
    // Define parameters
    int rank = 1;                      // The rank of the fft (1 = fft)
    int n[1] = {(int)nt};              // Dimensions
    int howmany = (int)N;              // Number of fft2 to be computed
    int *inembed = NULL;               // NULL is equivalent to passing n
    int istride = 1;                   // Distance between two elements in the input
    int idist = (int)nt;               // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;               // NULL is equivalent to passing n
    int ostride = 1;                   // Distance between two elements in the output
    int odist = (int)nt;               // Distance between k-th and (k+1)-th output elements
    unsigned int flags = FFTW_MEASURE; // bitwise OR ('|') of zero or more planner flags (see http://www.fftw.org/fftw3.pdf)

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
                                        FFTW_BACKWARD,
                                        flags);
    return plan;
}