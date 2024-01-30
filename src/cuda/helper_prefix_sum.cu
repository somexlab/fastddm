// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file helper_prefix_sum.cu
    \brief Definition of helper functions for cumulative sum (prefix sum) on GPU
*/

// *** headers ***
#include "helper_prefix_sum.cuh"
#include "helper_debug.cuh"

#include <cuda_runtime.h>

// *** code ***

/*!
    Compute next power of two larger or equal to n
 */
unsigned long long nextPowerOfTwo(unsigned long long n)
{
    unsigned long long power = 1;
    while (power < n)
    {
        power *= 2;
    }
    return power;
}
