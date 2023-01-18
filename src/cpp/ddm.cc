// Maintainer: enrico-lattuada

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

// *** code ***

/*!
    Compute the image structure function in diff mode
    using differences of Fourier transformed images.
 */
template <typename T>
py::array_t<double> ddm_diff(py::array_t<T, py::array::c_style> img_seq,
                             vector<unsigned int> lags,
                             unsigned long long nx,
                             unsigned long long ny)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();                // get input data

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      double [the input needs to be twice as large]
     */
    unsigned long long _nx = nx / 2 + 1;
    py::array_t<double> out = py::array_t<double>(2 * _nx * ny * (length + 2));
    auto p_out = out.mutable_data();

    // ***Create the fft2 plan
    fftw_plan fft2_plan = fft2_create_plan(p_out,
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

    // ***Execute fft2 plan
    fftw_execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) for Parseval theorem
    double norm_fact = sqrt((double)(nx * ny));
    for (unsigned long long ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        p_out[ii] /= norm_fact;
    }

    // ***Cleanup fft2 plan
    fftw_destroy_plan(fft2_plan);
    fftw_cleanup();

    // ***Compute the image structure function
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

        // normalize power spectrum
        tmp[lags.size()] /= (double)length;

        // get minus square modulus of average
        tmp[lags.size() + 1] /= (double)length;
        tmp2 /= (double)length;
        tmp[lags.size() + 1] *= -1 * tmp[lags.size() + 1];
        tmp[lags.size() + 1] -= tmp2 * tmp2;

        // compute variance
        tmp[lags.size() + 1] += tmp[lags.size()];

        // copy the values back in the vector
        copy_vec_with_stride(tmp,
                             p_out,
                             2 * q,
                             2 * (_nx * ny));
    }

    // Convert raw output to full and shifted image structure function
    make_full_shifted_isf(p_out,
                          nx,
                          ny,
                          lags.size() + 2);

    // Cleanup before finish
    tmp.clear();
    tmp.shrink_to_fit();

    // the full size of the image structure function is
    // nx * ny * #(lags)
    out.resize({(unsigned long long)(lags.size() + 2), ny, nx});

    // Return result to python
    return out;
}

/*!
    Compute the image structure function in fft mode
    using the Wiener-Khinchin theorem.

    Only (chunk_size) fft's in the t direction are computed
    simultaneously as a tradeoff between memory consumption
    and execution speed.

    Notice that nt must be at least 2*length to avoid
    circular correlation.
 */
template <typename T>
py::array_t<double> ddm_fft(py::array_t<T, py::array::c_style> img_seq,
                            vector<unsigned int> lags,
                            unsigned long long nx,
                            unsigned long long ny,
                            unsigned long long nt,
                            unsigned long long chunk_size)
{
    // ***Get input array and dimensions
    unsigned long long length = img_seq.shape()[0]; // get length of original input
    unsigned long long height = img_seq.shape()[1]; // get height of original input
    unsigned long long width = img_seq.shape()[2];  // get width of original input
    auto p_img_seq = img_seq.data();                // get input data

    // ***Allocate workspace vector
    /*
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      doubles [the input needs to be twice as large]
    - workspace will contain complex values, so we need 2* the size
      (allocated after fft2 part)
     */
    unsigned long long _nx = nx / 2 + 1;
    py::array_t<double> out = py::array_t<double>(2 * _nx * ny * length);
    auto p_out = out.mutable_data();

    // ***Create the fft2 plan
    fftw_plan fft2_plan = fft2_create_plan(p_out,
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

    // ***Execute fft2 plan
    fftw_execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) for Parseval theorem
    double norm_fact = sqrt((double)(nx * ny));
    for (unsigned long long ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        p_out[ii] /= norm_fact;
    }

    // ***Cleanup fft2 plan
    fftw_destroy_plan(fft2_plan);
    fftw_cleanup();

    // ***Allocate workspace
    vector<double> workspace(2 * chunk_size * nt);

    // ***Create the fft plan
    fftw_plan fft_plan = fft_create_plan(workspace,
                                         nt,
                                         chunk_size);

    // ***Compute the image structure function
    // initialize helper vector used in average part
    vector<double> tmp(chunk_size);
    for (unsigned long long i = 0; i < (_nx * ny - 1) / chunk_size + 1; i++)
    {
        // Step1: correlation part
        // copy values to workspace2 for fft
        for (unsigned long long q = 0; q < chunk_size; q++)
        {
            for (unsigned long long t = 0; t < length; t++)
            {
                workspace[2 * (q * nt + t)] = p_out[2 * (t * _nx * ny + i * chunk_size + q)];         // real
                workspace[2 * (q * nt + t) + 1] = p_out[2 * (t * _nx * ny + i * chunk_size + q) + 1]; // imag
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
            workspace[2 * j] = workspace[2 * j] * workspace[2 * j] + workspace[2 * j + 1] * workspace[2 * j + 1]; // real
            workspace[2 * j + 1] = 0.0;                                                                           // imag
        }

        // compute ifft
        fftw_execute(fft_plan);

        // Step2: average part
        unsigned long long idx = lags.size() - 1;
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
            if (length - t - 1 == lags[idx])
            {
                for (unsigned long long q = 0; q < chunk_size; ++q)
                {
                    // also divide corr part by nt to normalize fft
                    workspace[2 * (q * nt + lags[idx])] = tmp[q] - 2 * workspace[2 * (q * nt + lags[idx])] / (double)nt;
                    // finally, normalize output
                    workspace[2 * (q * nt + lags[idx])] /= (double)(length - lags[idx]);
                }
                if (idx == 0)
                {
                    fill(tmp.begin(), tmp.end(), 0.0);
                    break;
                }
                else
                {
                    idx--;
                }
            }
        }

        // Step3: copy results to workspace1
        for (unsigned long long idx = 0; idx < lags.size(); idx++)
        {
            for (unsigned long long q = 0; q < chunk_size; q++)
            {
                p_out[2 * (idx * _nx * ny + i * chunk_size + q)] = workspace[2 * (q * nt + lags[idx])];
            }
        }
    }

    // Convert raw output to full and shifted image structure function
    make_full_shifted_isf(p_out,
                          nx,
                          ny,
                          lags.size());

    // Cleanup before finish
    fftw_destroy_plan(fft_plan);
    fftw_cleanup();
    workspace.clear();
    workspace.shrink_to_fit();
    tmp.clear();
    tmp.shrink_to_fit();

    // the full size of the image structure function is
    // nx * ny * #(lags)
    out.resize({(unsigned long long)(lags.size()), ny, nx});

    // Return result to python
    return out;
}

/*!
    Export ddm functions to python.
 */
void export_ddm(py::module &m)
{
    // Leave function export in this order!
    m.def("ddm_diff", &ddm_diff<uint8_t>);
    m.def("ddm_diff", &ddm_diff<int16_t>);
    m.def("ddm_diff", &ddm_diff<uint16_t>);
    m.def("ddm_diff", &ddm_diff<int32_t>);
    m.def("ddm_diff", &ddm_diff<uint32_t>);
    m.def("ddm_diff", &ddm_diff<int64_t>);
    m.def("ddm_diff", &ddm_diff<uint64_t>);
    m.def("ddm_diff", &ddm_diff<float>);
    m.def("ddm_diff", &ddm_diff<double>);
    m.def("ddm_fft", &ddm_fft<uint8_t>);
    m.def("ddm_fft", &ddm_fft<int16_t>);
    m.def("ddm_fft", &ddm_fft<uint16_t>);
    m.def("ddm_fft", &ddm_fft<int32_t>);
    m.def("ddm_fft", &ddm_fft<uint32_t>);
    m.def("ddm_fft", &ddm_fft<int64_t>);
    m.def("ddm_fft", &ddm_fft<uint64_t>);
    m.def("ddm_fft", &ddm_fft<float>);
    m.def("ddm_fft", &ddm_fft<double>);
}
