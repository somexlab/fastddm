// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __DATA_STRUCT_H__
#define __DATA_STRUCT_H__

/*! \file data_struct.h
    \brief Defines data structures to hold variables for the different parts of the calculation
*/

// *** headers ***

// *** code ***

/*! \brief Data structure to hold image sequence parameters
 */
struct ImageData
{
    unsigned long long length;               // Number of frames in the sequence
    unsigned long long height;               // Height of each frame in the sequence
    unsigned long long width;                // Width of each frame in the sequence
    bool is_input_type_scalar;               // True if the input type is Scalar
    unsigned long long input_type_num_bytes; // Number of bytes in the input type
};

/*! \brief Data structure to hold structure function parameters
 */
struct StructureFunctionData
{
    unsigned long long nx;       // Number of grid points in x
    unsigned long long ny;       // Number of grid points in y
    unsigned long long num_lags; // Number of lags analyzed
    unsigned long long length;   // Total length of the structure function (includes power spectrum and variance)
    unsigned long long nx_half;  // Number of grid points of the half-plane representation of the real-to-complex FFT2
    bool is_window;              // True if the window is applied
};

/*! \brief Data structure to hold the execution parameters
 */
struct ExecutionParameters
{
    unsigned long long nt;                 // Number of grid points in t (used in "fft" mode)
    unsigned long long num_fft2_loops;     // Number of batched FFT2 loops
    unsigned long long num_batch_loops;    // Number of batched loops
    unsigned long long num_fftshift_loops; // Number of batched FFTshift loops
};

/*! \brief Data structure to hold the pitch parameters for efficient memory transfer
 */
struct PitchData
{
    unsigned long long p_buffer;   // Pitch of the buffer
    unsigned long long p_nx;       // Pitch of the nx array
    unsigned long long p_q;        // Pitch of the q array
    unsigned long long p_t;        // Pitch of the t array
    unsigned long long p_nt;       // Pitch of the nt array
    unsigned long long p_fftshift; // Pitch of the fftshift array
};

#endif // __DATA_STRUCT_H__
