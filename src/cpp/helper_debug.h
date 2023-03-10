// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

#ifndef __HELPER_DEBUG_H__
#define __HELPER_DEBUG_H__

#include <vector>
#include <iostream>

using namespace std;

void display_vector(vector<double> *vec,
                    size_t nx,
                    size_t ny,
                    size_t nz)
{
    for (size_t t = 0; t < nz; t++)
    {
        for (size_t y = 0; y < ny; y++)
        {
            for (size_t x = 0; x < nx; x++)
            {
                cout << (*vec)[t * (nx * ny) + y * nx + x] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

void display_array(double *vec,
                   size_t nx,
                   size_t ny,
                   size_t nz)
{
    for (size_t t = 0; t < nz; t++)
    {
        for (size_t y = 0; y < ny; y++)
        {
            for (size_t x = 0; x < nx; x++)
            {
                cout << vec[t * (nx * ny) + y * nx + x] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }
}

#endif  // __HELPER_DEBUG_H__