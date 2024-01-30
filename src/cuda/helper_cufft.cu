// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file helper_cufft.cu
    \brief Definition of helper functions for cufft execution on GPU
*/

// *** headers ***
#include "helper_cufft.cuh"
#include "gpu_utils.cuh"
#include "helper_debug.cuh"

// *** code ***

/*!
    Get the memory size needed for the work area for a 2D cufft
*/
unsigned long long get_fft2_device_memory_size(size_t nx,
                                               size_t ny,
                                               size_t batch,
                                               size_t pitch,
                                               cufftResult &cufft_res)
{
    // Initialize array for memory sizes
    size_t memsize[1];

    // Define cufft2 plan parameters
    int rank = 2;                               // The rank of the fft (2 = fft2)
    int n[2] = {(int)ny, (int)nx};              // Dimensions
    int inembed[2] = {(int)ny, 2 * (int)pitch}; // nembed input values (NULL is equivalent to passing n)
    int istride = 1;                            // Distance between two elements in the input
    int idist = (int)(2 * ny * pitch);          // Distance between k-th and (k+1)-th input elements
    int onembed[2] = {(int)ny, (int)pitch};     // nembed output values (NULL is equivalent to passing n)
    int ostride = 1;                            // Distance between two elements in the output
    int odist = (int)(ny * pitch);              // Distance between k-th and (k+1)-th output elements
    cufftType type = CUFFT_D2Z;                 // FFT type (real to complex, double precision)

    // Create the fft2 plan
    cufftHandle plan;
    cufftSilentSafeCall(cufftPlanMany(&plan,
                                      rank,
                                      n,
                                      inembed,
                                      istride,
                                      idist,
                                      onembed,
                                      ostride,
                                      odist,
                                      type,
                                      (int)batch));

    // Evaluate the memory size
    cufft_res = cufftGetSizeMany(plan,
                                 rank,
                                 n,
                                 inembed,
                                 istride,
                                 idist,
                                 onembed,
                                 ostride,
                                 odist,
                                 type,
                                 (int)batch,
                                 memsize);

    cufftSilentSafeCall(cufftDestroy(plan));

    return (unsigned long long)memsize[0];
}

/*!
    Create the cufft plan for the real to complex fft2
*/
cufftHandle create_fft2_plan(size_t nx,
                             size_t ny,
                             size_t batch,
                             size_t pitch)
{
    // Define parameters
    int rank = 2;                               // The rank of the fft (2 = fft2)
    int n[2] = {(int)ny, (int)nx};              // Dimensions
    int inembed[2] = {(int)ny, 2 * (int)pitch}; // nembed input values (NULL is equivalent to passing n)
    int istride = 1;                            // Distance between two elements in the input
    int idist = (int)(2 * ny * pitch);          // Distance between k-th and (k+1)-th input elements
    int onembed[2] = {(int)ny, (int)pitch};     // nembed output values (NULL is equivalent to passing n)
    int ostride = 1;                            // Distance between two elements in the output
    int odist = (int)(ny * pitch);              // Distance between k-th and (k+1)-th output elements
    cufftType type = CUFFT_D2Z;                 // FFT type (real to complex, double precision)

    // Create the fft2 plan
    cufftHandle plan;
    cufftSafeCall(cufftPlanMany(&plan,
                                rank,
                                n,
                                inembed,
                                istride,
                                idist,
                                onembed,
                                ostride,
                                odist,
                                type,
                                (int)batch));

    return plan;
}

/*!
    Get the memory size needed for the work area for a 1D cufft
*/
unsigned long long get_fft_device_memory_size(size_t nt,
                                              size_t batch,
                                              size_t pitch,
                                              cufftResult &cufft_res)
{
    // Initialize array for memory sizes
    size_t memsize[1];

    // Define cufft plan parameters
    int rank = 1;               // The rank of the fft (1 = fft)
    int n[1] = {(int)nt};       // Dimensions
    int *inembed = NULL;        // nembed input values (NULL is equivalent to passing n)
    int istride = 1;            // Distance between two elements in the input
    int idist = (int)(pitch);   // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;        // nembed output values (NULL is equivalent to passing n)
    int ostride = 1;            // Distance between two elements in the output
    int odist = (int)pitch;     // Distance between k-th and (k+1)-th output elements
    cufftType type = CUFFT_Z2Z; // FFT type (complex to complex, double precision)

    // Create the fft2 plan
    cufftHandle plan;
    cufftSilentSafeCall(cufftPlanMany(&plan,
                                      rank,
                                      n,
                                      inembed,
                                      istride,
                                      idist,
                                      onembed,
                                      ostride,
                                      odist,
                                      type,
                                      (int)batch));

    // Evaluate the memory size
    cufft_res = cufftGetSizeMany(plan,
                                 rank,
                                 n,
                                 inembed,
                                 istride,
                                 idist,
                                 onembed,
                                 ostride,
                                 odist,
                                 type,
                                 (int)batch,
                                 memsize);

    cufftSilentSafeCall(cufftDestroy(plan));

    return (unsigned long long)memsize[0];
}

/*!
    Create the cufft plan for the complex to complex fft
*/
cufftHandle create_fft_plan(size_t nt,
                            size_t batch,
                            size_t pitch)
{
    // Define cufft plan parameters
    int rank = 1;                  // The rank of the fft (1 = fft)
    int n[1] = {(int)nt};          // Dimensions
    int inembed[1] = {(int)pitch}; // nembed input values (NULL is equivalent to passing n)
    int istride = 1;               // Distance between two elements in the input
    int idist = (int)(pitch);      // Distance between k-th and (k+1)-th input elements
    int onembed[1] = {(int)pitch}; // nembed output values (NULL is equivalent to passing n)
    int ostride = 1;               // Distance between two elements in the output
    int odist = (int)pitch;        // Distance between k-th and (k+1)-th output elements
    cufftType type = CUFFT_Z2Z;    // FFT type (complex to complex, double precision)

    // Create the fft plan
    cufftHandle plan;
    cufftSafeCall(cufftPlanMany(&plan,
                                rank,
                                n,
                                inembed,
                                istride,
                                idist,
                                onembed,
                                ostride,
                                odist,
                                type,
                                (int)batch));

    return plan;
}
