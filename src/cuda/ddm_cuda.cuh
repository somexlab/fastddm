// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __DDM_CUDA_CUH__
#define __DDM_CUDA_CUH__

/*! \file ddm_cuda.cuh
    \brief Declaration of core Differential Dynamic Microscopy functions for GPU
*/

// *** headers ***
#include "data_struct.h"

#include <vector>

using namespace std;

#ifndef SINGLE_PRECISION
typedef double Scalar;
#else
typedef float Scalar;
#endif

// *** code ***

/*! \brief Transfer images on GPU and compute fft2
    \param h_in         Input array
    \param h_out        Output array
    \param h_window     Window function
    \param img_data     Structure holding the image sequence parameters
    \param sf_data      Structure holding the structure function parameters
    \param exec_params  Structure holding the execution parameters
    \param pitch_data   Structure holding the memory pitch parameters
 */
template <typename T>
void compute_fft2(const T *h_in,
                  Scalar *h_out,
                  const Scalar *h_window,
                  ImageData &img_data,
                  StructureFunctionData &sf_data,
                  ExecutionParameters &exec_params,
                  PitchData &pitch_data);

/*! \brief Compute structure function using differences on the GPU
    \param h_in         Input array of Fourier transformed images
    \param lags         Lags to be analyzed
    \param img_data     Structure holding the image sequence parameters
    \param sf_data      Structure holding the structure function parameters
    \param exec_params  Structure holding the execution parameters
    \param pitch_data   Structure holding the memory pitch parameters
 */
void structure_function_diff(Scalar *h_in,
                             vector<unsigned int> lags,
                             ImageData &img_data,
                             StructureFunctionData &sf_data,
                             ExecutionParameters &exec_params,
                             PitchData &pitch_data);

/*! \brief Compute structure function using the Wiener-Khinchin theorem on the GPU
    \param h_in         Input array of Fourier transformed images
    \param lags         Lags to be analyzed
    \param img_data     Structure holding the image sequence parameters
    \param sf_data      Structure holding the structure function parameters
    \param exec_params  Structure holding the execution parameters
    \param pitch_data   Structure holding the memory pitch parameters
 */
void structure_function_fft(Scalar *h_in,
                            vector<unsigned int> lags,
                            ImageData &img_data,
                            StructureFunctionData &sf_data,
                            ExecutionParameters &exec_params,
                            PitchData &pitch_data);

/*! \brief Convert to fftshifted structure function on the GPU
    \param h_in         Input array after structure function calculation
    \param img_data     Structure holding the image sequence parameters
    \param sf_data      Structure holding the structure function parameters
    \param exec_params  Structure holding the execution parameters
    \param pitch_data   Structure holding the memory pitch parameters
 */
void make_shift(Scalar *h_in,
                ImageData &img_data,
                StructureFunctionData &sf_data,
                ExecutionParameters &exec_params,
                PitchData &pitch_data);

#endif // __DDM_CUDA_CUH__
