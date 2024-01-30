// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

// inclusion guard
#ifndef __HELPER_PREFIX_SUM_CUH__
#define __HELPER_PREFIX_SUM_CUH__

/*! \file helper_prefix_sum.cuh
    \brief Declaration of helper functions for cumulative sum (prefix sum) on GPU
*/

// *** headers ***

// *** code ***

/*! \brief Compute next power of two larger or equal to n
    \param n    Target
 */
unsigned long long nextPowerOfTwo(unsigned long long n);

#endif // __HELPER_PREFIX_SUM_CUH__
