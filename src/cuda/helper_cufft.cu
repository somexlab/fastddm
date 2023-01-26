// Maintainer: enrico-lattuada

/*! \file helper_cufft.cu
    \brief Definition of cufft helper functions
*/

// *** headers ***
#include "helper_cufft.cuh"
#include "helper_debug.cuh"

// *** code ***

/*
    Create the cufft plan for the fft2.
 */
cufftHandle fft2_create_plan(size_t nx,
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
    cufftType type = CUFFT_D2Z;                 // Fft type (real to complex, double precision, here)

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
    Evaluate the device memory size in bytes for fft2
 */
void fft2_get_mem_size(size_t nx,
                       size_t ny,
                       size_t batch,
                       size_t pitch,
                       size_t *memsize,
                       cufftResult &cufft_res)
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
    cufftType type = CUFFT_D2Z;                 // Fft type (real to complex, double precision, here)

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
}

/*
    Create the cufft plan for the complex to complex fft.
 */
cufftHandle fft_create_plan(size_t nt,
                            size_t batch,
                            size_t pitch)
{
    // Define parameters
    int rank = 1;                 // The rank of the fft (1 = fft)
    int n[1] = {(int)nt};         // Dimensions
    int inembed[] = {(int)pitch}; // NULL is equivalent to passing n
    int istride = 1;              // Distance between two elements in the input
    int idist = (int)pitch;       // Distance between k-th and (k+1)-th input elements
    int onembed[] = {(int)pitch}; // NULL is equivalent to passing n
    int ostride = 1;              // Distance between two elements in the output
    int odist = (int)pitch;       // Distance between k-th and (k+1)-th output elements
    cufftType type = CUFFT_Z2Z;   // Fft type (complex to complex, double precision, here)

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

/*!
    Evaluate the device memory size in bytes for fft
 */
void fft_get_mem_size(size_t nt,
                      size_t batch,
                      size_t pitch,
                      size_t *memsize,
                      cufftResult &cufft_res)
{
    // Define parameters
    int rank = 1;               // The rank of the fft (1 = fft)
    int n[1] = {(int)nt};       // Dimensions
    int *inembed = NULL;        // NULL is equivalent to passing n
    int istride = 1;            // Distance between two elements in the input
    int idist = (int)pitch;     // Distance between k-th and (k+1)-th input elements
    int *onembed = NULL;        // NULL is equivalent to passing n
    int ostride = 1;            // Distance between two elements in the output
    int odist = (int)pitch;     // Distance between k-th and (k+1)-th output elements
    cufftType type = CUFFT_Z2Z; // Fft type (complex to complex, double precision, here)

    // Create the fft plan
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
}
