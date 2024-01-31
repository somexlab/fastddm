// Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
// Part of FastDDM, released under the GNU GPL-3.0 License.

// Author: Enrico Lattuada
// Maintainer: Enrico Lattuada

/*! \file ddm.cc
    \brief Definition of core C++ Differential Dynamic Microscopy functions
*/

// *** headers ***
#include "ddm.h"

#include "helper_fftw.h"
#include "helper_ddm.h"
#include "helper_debug.h"

#include <chrono>
using namespace std::chrono;

#ifndef SINGLE_PRECISION
void (*Fftw_Execute)(fftw_plan) = &fftw_execute;
void (*Fftw_Destroy_Plan)(fftw_plan) = &fftw_destroy_plan;
void (*Fftw_Cleanup)() = &fftw_cleanup;
#else
void (*Fftw_Execute)(fftwf_plan) = &fftwf_execute;
void (*Fftw_Destroy_Plan)(fftwf_plan) = &fftwf_destroy_plan;
void (*Fftw_Cleanup)() = &fftwf_cleanup;
#endif

// *** code ***

/*!
    Compute the structure function in "diff" mode
    using differences of Fourier transformed images.
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<T, py::array::c_style> img_seq,
                                             vector<unsigned int> lags,
                                             unsigned long long nx,
                                             unsigned long long ny,
                                             py::array_t<Scalar, py::array::c_style> window)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();                // get input data

    // ***Get window array
    unsigned long long window_length = window.shape()[0]; // get length of window array
    auto p_window = window.data();

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      Scalar [the input needs to be twice as large]
     */
    unsigned long long _nx = nx / 2 + 1;
    unsigned long long dim_t = max(length, (unsigned long long)(lags.size() + 2));
    py::array_t<Scalar> out = py::array_t<Scalar>(2 * _nx * ny * dim_t);
    auto p_out = out.mutable_data();

    // ***Create the fft2 plan
    FFTW_PLAN fft2_plan = fft2_create_plan(p_out,
                                           nx,
                                           ny,
                                           length);

    // ***Copy input to workspace vector
    for (unsigned long long t = 0; t < length; t++)
    {
        for (unsigned long long y = 0; y < height; y++)
        {
            copy(p_img_seq + t * (height * width) + y * width,
                 p_img_seq + t * (height * width) + (y + 1) * width,
                 p_out + t * (2 * _nx * ny) + y * 2 * _nx);
        }
    }

    // ***Check if window function is not empty, if so, apply window
    if (window_length > 0)
    {
        for (unsigned long long t = 0; t < length; t++)
        {
            for (unsigned long long y = 0; y < height; y++)
            {
                for (unsigned long long x = 0; x < width; x++)
                {
                    p_out[t * (2 * _nx * ny) + y * 2 * _nx + x] *= p_window[y * width + x];
                }
            }
        }
    }

    // ***Execute fft2 plan
    Fftw_Execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) for Parseval theorem
    Scalar norm_fact = sqrt((double)(nx * ny));
    for (unsigned long long ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        p_out[ii] /= norm_fact;
    }

    // ***Cleanup fft2 plan
    Fftw_Destroy_Plan(fft2_plan);
    Fftw_Cleanup();

    // ***Compute the structure function
    // initialize helper vector
    vector<double> tmp(lags.size() + 2, 0.0);
    double tmp2 = 0.0;

    // loop over the q values
    for (unsigned long long q = 0; q < _nx * ny; q++)
    {
        // zero out the helper vector
        fill(tmp.begin(), tmp.end(), 0);
        tmp2 = 0.0;

        // loop over the lags
        for (unsigned long long _dt = 0; _dt < lags.size(); _dt++)
        {
            // get current lag
            unsigned long long dt = lags[_dt];

            // loop over time
            for (unsigned long long t = 0; t < length - dt; t++)
            {
                // compute the power spectrum of the difference of pixel at time t and time t+dt, i.e.
                // [(a+ib) - (c+id)] * conj[(a+ib) - (c+id)] = (a-c)^2+(b-d)^2
                // notice fft is complex, so the stride between two consecutive pixels is two
                double a = p_out[2 * ((t + dt) * (_nx * ny) + q)];
                double b = p_out[2 * ((t + dt) * (_nx * ny) + q) + 1];
                double c = p_out[2 * ((t) * (_nx * ny) + q)];
                double d = p_out[2 * ((t) * (_nx * ny) + q) + 1];

                tmp[_dt] += (a - c) * (a - c) + (b - d) * (b - d);
            }

            // normalize
            tmp[_dt] /= (double)(length - dt);
        }

        // compute average power spectrum and variance
        for (unsigned long long t = 0; t < length; t++)
        {
            double a = p_out[2 * ((t) * (_nx * ny) + q)];
            double b = p_out[2 * ((t) * (_nx * ny) + q) + 1];
            tmp[lags.size()] += a * a + b * b;
            tmp[lags.size() + 1] += a;
            tmp2 += b;
        }

        // power spectrum
        tmp[lags.size()] /= (double)length;

        // variance
        tmp[lags.size() + 1] /= (double)length;
        tmp2 /= (double)length;
        tmp[lags.size() + 1] = tmp[lags.size()] - tmp[lags.size() + 1] * tmp[lags.size() + 1] - tmp2 * tmp2;

        // copy the values back in the vector
        copy_vec_with_stride(tmp,
                             p_out,
                             2 * q,
                             2 * (_nx * ny));
    }

    // Convert raw output to shifted structure function
    make_shifted_isf(p_out,
                     nx,
                     ny,
                     lags.size() + 2);

    // Cleanup before finish
    tmp.clear();
    tmp.shrink_to_fit();

    // the size of the half-plane structure function is
    // (nx / 2 + 1) * ny * [#(lags) + 2]
    out.resize({(unsigned long long)(lags.size() + 2), ny, _nx});

    // release pointer to output array
    p_out = NULL;

    // Return result to python
    return out;
}
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<uint8_t, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<int16_t, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<uint16_t, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<int32_t, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<uint32_t, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<int64_t, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<uint64_t, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<float, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_diff(py::array_t<double, py::array::c_style> img_seq,
                                                      vector<unsigned int> lags,
                                                      unsigned long long nx,
                                                      unsigned long long ny,
                                                      py::array_t<Scalar, py::array::c_style> window);

/*!
    Compute the structure function in "fft" mode
    using the Wiener-Khinchin theorem.

    Only (chunk_size) fft's in the t direction are computed
    simultaneously as a tradeoff between memory consumption
    and execution speed.

    Notice that nt must be at least 2*length to avoid
    circular correlation.
 */
template <typename T>
py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<T, py::array::c_style> img_seq,
                                            vector<unsigned int> lags,
                                            unsigned long long nx,
                                            unsigned long long ny,
                                            unsigned long long nt,
                                            unsigned long long chunk_size,
                                            py::array_t<Scalar, py::array::c_style> window)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();                // get input data

    // ***Get window array
    unsigned long long window_length = window.shape()[0]; // get length of window array
    auto p_window = window.data();

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      Scalar [the input needs to be twice as large]
    - workspace will contain complex values, so we need 2* the size
      (allocated after fft2 part)
     */
    unsigned long long _nx = nx / 2 + 1;
    unsigned long long dim_t = max(length, (unsigned long long)(lags.size() + 2));
    py::array_t<Scalar> out = py::array_t<Scalar>(2 * _nx * ny * dim_t);
    auto p_out = out.mutable_data();

    // ***Create the fft2 plan
    FFTW_PLAN fft2_plan = fft2_create_plan(p_out,
                                           nx,
                                           ny,
                                           length);

    // ***Copy input to workspace vector
    for (unsigned long long t = 0; t < length; t++)
    {
        for (unsigned long long y = 0; y < height; y++)
        {
            copy(p_img_seq + t * (height * width) + y * width,
                 p_img_seq + t * (height * width) + (y + 1) * width,
                 p_out + t * (2 * _nx * ny) + y * 2 * _nx);
        }
    }

    // ***Check if window function is not empty, if so, apply window
    if (window_length > 0)
    {
        for (unsigned long long t = 0; t < length; t++)
        {
            for (unsigned long long y = 0; y < height; y++)
            {
                for (unsigned long long x = 0; x < width; x++)
                {
                    p_out[t * (2 * _nx * ny) + y * 2 * _nx + x] *= p_window[y * width + x];
                }
            }
        }
    }

    // ***Execute fft2 plan
    Fftw_Execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) for Parseval theorem
    Scalar norm_fact = sqrt((double)(nx * ny));
    for (unsigned long long ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        p_out[ii] /= norm_fact;
    }

    // ***Cleanup fft2 plan
    Fftw_Destroy_Plan(fft2_plan);
    Fftw_Cleanup();

    // ***Allocate workspace
    vector<double> workspace(2 * chunk_size * nt);

    // ***Create the fft plan
    fftw_plan fft_plan = fft_create_plan(workspace,
                                         nt,
                                         chunk_size);

    // ***Compute the structure function
    // initialize helper vector used in average part
    vector<double> tmp(chunk_size);
    // initialize helper vector used in square modulus of average Fourier transform
    vector<double> tmpAvg(chunk_size);
    // normalize fft before actually computing it
    double norm_fact_2 = sqrt((double)(nt));
    for (unsigned long long i = 0; i < (_nx * ny - 1) / chunk_size + 1; i++)
    {
        // Step1: correlation part
        // copy values to workspace2 for fft
        for (unsigned long long q = 0; q < chunk_size; q++)
        {
            for (unsigned long long t = 0; t < length; t++)
            {
                workspace[2 * (q * nt + t)] = p_out[2 * (t * _nx * ny + i * chunk_size + q)] / norm_fact_2;         // real
                workspace[2 * (q * nt + t) + 1] = p_out[2 * (t * _nx * ny + i * chunk_size + q) + 1] / norm_fact_2; // imag
            }
            // set other values to 0
            for (unsigned long long t = length; t < nt; t++)
            {
                workspace[2 * (q * nt + t)] = 0.0;
                workspace[2 * (q * nt + t) + 1] = 0.0;
            }
        }

        // compute the fft
        fftw_execute(fft_plan);

        // compute power spectrum of fft
        for (unsigned long long j = 0; j < chunk_size * nt; j++)
        {
            double a = workspace[2 * j];
            double b = workspace[2 * j + 1];
            workspace[2 * j] = a * a + b * b; // real
            workspace[2 * j + 1] = 0.0;       // imag
        }

        // copy value in 0
        for (unsigned long long q = 0; q < chunk_size; q++)
        {
            // compute average
            // Take into account pre-normalization of the fft and actual normalization factor
            tmpAvg[q] = workspace[2 * q * nt] * (double)nt / (double)(length * length);
            // Initialize tmp to zero
            tmp[q] = 0.0;
        }

        // compute ifft
        fftw_execute(fft_plan);

        // Step2: average part
        unsigned long long idx = 0;
        for (unsigned long long t = 0; t < length; t++)
        {
            for (unsigned long long q = 0; q < chunk_size; q++)
            {
                double a = p_out[2 * (t * _nx * ny + i * chunk_size + q)];     // real
                double b = p_out[2 * (t * _nx * ny + i * chunk_size + q) + 1]; // imag
                tmp[q] += a * a + b * b;
                a = p_out[2 * ((length - t - 1) * _nx * ny + i * chunk_size + q)];     // real
                b = p_out[2 * ((length - t - 1) * _nx * ny + i * chunk_size + q) + 1]; // imag
                tmp[q] += a * a + b * b;
            }

            // add contribution only if delay in list
            if ((idx < lags.size()) && (length - t - 1 == lags[lags.size() - 1 - idx]))
            {
                for (unsigned long long q = 0; q < chunk_size; ++q)
                {
                    // also divide corr part by nt to normalize fft
                    workspace[2 * (q * nt + lags[lags.size() - 1 - idx])] = tmp[q] - 2.0 * workspace[2 * (q * nt + lags[lags.size() - 1 - idx])]; // / (double)nt;
                    // finally, normalize output
                    workspace[2 * (q * nt + lags[lags.size() - 1 - idx])] /= (double)(length - lags[lags.size() - 1 - idx]);
                }
                idx++;
            }
        }

        // Step3: copy results to workspace1
        for (unsigned long long q = 0; q < chunk_size; q++)
        {
            for (unsigned long long idx = 0; idx < lags.size(); idx++)
            {
                p_out[2 * (idx * _nx * ny + i * chunk_size + q)] = workspace[2 * (q * nt + lags[idx])];
            }
            p_out[2 * (lags.size() * _nx * ny + i * chunk_size + q)] = 0.5 * tmp[q] / (double)length;
            p_out[2 * ((lags.size() + 1) * _nx * ny + i * chunk_size + q)] = p_out[2 * (lags.size() * _nx * ny + i * chunk_size + q)] - tmpAvg[q];
        }
    }

    // Convert raw output to shifted structure function
    make_shifted_isf(p_out,
                     nx,
                     ny,
                     lags.size() + 2);

    // Cleanup before finish
    fftw_destroy_plan(fft_plan);
    fftw_cleanup();
    workspace.clear();
    workspace.shrink_to_fit();
    tmp.clear();
    tmp.shrink_to_fit();
    tmpAvg.clear();
    tmpAvg.shrink_to_fit();

    // the size of the half-plane structure function is
    // (nx / 2 + 1) * ny * [#(lags) + 2]
    out.resize({(unsigned long long)(lags.size() + 2), ny, _nx});

    // release pointer to output array
    p_out = NULL;

    // Return result to python
    return out;
}
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<uint8_t, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<int16_t, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<uint16_t, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<int32_t, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<uint32_t, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<int64_t, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<uint64_t, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<float, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
template py::array_t<Scalar> PYBIND11_EXPORT ddm_fft(py::array_t<double, py::array::c_style> img_seq,
                                                     vector<unsigned int> lags,
                                                     unsigned long long nx,
                                                     unsigned long long ny,
                                                     unsigned long long nt,
                                                     unsigned long long chunk_size,
                                                     py::array_t<Scalar, py::array::c_style> window);
