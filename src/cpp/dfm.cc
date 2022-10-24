// Maintainer: enrico-lattuada

/*! \file dfm.cc
    \brief Definition of core C++ Digital Fourier Microscopy functions
*/

// *** headers ***
#include "dfm.h"
#include "pycomm.h"

#include "helper_fftw.h"
#include "helper_dfm.h"
#include "helper_debug.h"

#include <chrono>
using namespace std::chrono;

// *** code ***

/*!
    Compute the image structure function in direct mode
    using differences of fourier transformed images.
 */
template <typename T>
py::array_t<double> dfm_direct(py::array_t<T, py::array::c_style> img_seq,
                               vector<unsigned int> lags,
                               size_t nx,
                               size_t ny)
{
    string _logger = "";

    // ***Get input array dimensions
    auto buff = img_seq.request(); // get pointer to values
    size_t length = buff.shape[0]; // get length of original input
    size_t height = buff.shape[1]; // get height of original input
    size_t width = buff.shape[2];  // get width of original input

    // ***Allocate workspace vector
    /*
    - Vector needs to be allocated on heap so that on return
      `vector2numpy` can take ownership
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      doubles [the input needs to be twice as large]
     */
    auto t0 = high_resolution_clock::now(); // start

    size_t _nx = nx / 2 + 1;
    vector<double> *workspace = new vector<double>(2 * _nx * ny * length, 0.0);

    auto t1 = high_resolution_clock::now(); // stop
    duration<double> dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Workspace allocation (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Create the fft2 plan
    t0 = high_resolution_clock::now(); // start

    fftw_plan fft2_plan = fft2_create_plan(*workspace,
                                           nx,
                                           ny,
                                           length);

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "FFT2 plan creation (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Copy input to workspace vector
    t0 = high_resolution_clock::now(); // start

    for (size_t t = 0; t < length; t++)
    {
        for (size_t y = 0; y < height; y++)
        {
            copy(img_seq.data() + t * (height * width) + y * width,
                 img_seq.data() + t * (height * width) + (y + 1) * width,
                 workspace->begin() + t * (2 * _nx * ny) + y * 2 * _nx);
        }
    }

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Input copy (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Execute fft2 plan
    t0 = high_resolution_clock::now(); // start

    fftw_execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) to preserve Parseval theorem
    double norm_fact = sqrt((double)(nx * ny));
    for (size_t ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        (*workspace)[ii] /= norm_fact;
    }

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "FFT2 execution (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Compute the image structure function
    t0 = high_resolution_clock::now(); // start
    // initialize helper vector
    vector<double> tmp(lags.size(), 0.0);

    // loop over the q values
    for (size_t q = 0; q < _nx * ny; q++)
    {
        // zero out the helper vector
        fill(tmp.begin(), tmp.end(), 0);

        // loop over the lags
        for (size_t _dt = 0; _dt < lags.size(); _dt++)
        {
            // get current lag
            size_t dt = lags[_dt];

            // loop over time
            for (size_t t = 0; t < length - dt; t++)
            {
                // compute the power spectrum of the difference of pixel at time t and time t+dt, i.e.
                // [(a+ib) - (c+id)] * conj[(a+ib) - (c+id)] = (a-c)^2+(b-d)^2
                // notice fft is complex, so the stride between two consecutive pixels is two
                double a = (*workspace)[2 * ((t + dt) * (_nx * ny) + q)];
                double b = (*workspace)[2 * ((t + dt) * (_nx * ny) + q) + 1];
                double c = (*workspace)[2 * ((t) * (_nx * ny) + q)];
                double d = (*workspace)[2 * ((t) * (_nx * ny) + q) + 1];

                tmp[_dt] += (a - c) * (a - c) + (b - d) * (b - d);
            }

            // normalize
            tmp[_dt] /= (double)(length - dt);
        }

        // copy the values back in the vector
        copy_vec_with_stride(tmp,
                             *workspace,
                             2 * q,
                             2 * (_nx * ny));
    }

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "ISF calculation (s):\n" + to_string(dt.count()) + "\n\n";

    /*
    // Make full image structure function (keep real part and copy symmetric part)
    t0 = high_resolution_clock::now(); // start

    make_full_isf(*workspace,
                  nx,
                  ny,
                  lags.size());

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    logs += "Conversion to full ISF (s):\n" + to_string(dt.count()) + "\n\n";

    // FFTshift before output
    t0 = high_resolution_clock::now(); // start

    fft2_shift(*workspace,
               nx,
               ny,
               lags.size());

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    logs += "FFTshift ISF (s):\n" + to_string(dt.count()) + "\n\n";
     */

    // Convert raw output to full and shifted ISF
    t0 = high_resolution_clock::now(); // start

    make_full_shifted_isf(*workspace,
                          nx,
                          ny,
                          lags.size());

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Conversion to full ISF (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Shrink workspace if needed
    t0 = high_resolution_clock::now(); // start

    // the full size of the image structure function is
    // nx * ny * #(lags)
    workspace->resize(nx * ny * lags.size());
    workspace->shrink_to_fit();

    // Cleanup before finish
    fftw_destroy_plan(fft2_plan);
    fftw_cleanup();
    tmp.clear();
    tmp.shrink_to_fit();

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Garbage cleaning (s):\n" + to_string(dt.count()) + "\n\n";

    _logger.clear();

    // Return result to python
    return vector2numpy(workspace, nx, ny, lags.size());
}

/*!
    Compute the image structure function in fft mode
    using the Wiener-Khinchin theorem.

    Only (bundle_size) fft's in the t direction are computed
    simultaneously as a tradeoff between memory consumption
    and execution speed.

    Notice that nt must be at least 2*length to avoid
    circular correlation.
 */
template <typename T>
py::array_t<double> dfm_fft(py::array_t<T, py::array::c_style> img_seq,
                            vector<unsigned int> lags,
                            size_t nx,
                            size_t ny,
                            size_t nt,
                            size_t bundle_size)
{
    string _logger = "";

    // ***Get input array dimensions
    auto buff = img_seq.request(); // get pointer to values
    size_t length = buff.shape[0]; // get length of original input
    size_t height = buff.shape[1]; // get height of original input
    size_t width = buff.shape[2];  // get width of original input

    // ***Allocate workspace vector
    /*
    - Vector needs to be allocated on heap so that on return
      `vector2numpy` can take ownership
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      doubles [the input needs to be twice as large]
    - workspace2 will contain complex values, so we need 2* the size
     */
    auto t0 = high_resolution_clock::now(); // start

    size_t _nx = nx / 2 + 1;
    vector<double> *workspace1 = new vector<double>(2 * _nx * ny * length, 0.0);
    vector<double> *workspace2 = new vector<double>(2 * bundle_size * nt, 0.0);

    auto t1 = high_resolution_clock::now(); // stop
    duration<double> dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Workspace allocation (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Create the fft2 plan
    t0 = high_resolution_clock::now(); // start

    fftw_plan fft2_plan = fft2_create_plan(*workspace1,
                                           nx,
                                           ny,
                                           length);

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "FFT2 plan creation (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Create the fft and ifft plans
    t0 = high_resolution_clock::now(); // start

    fftw_plan fft_plan = fft_create_plan(*workspace2,
                                         nt,
                                         bundle_size);

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "FFT plan creation (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Copy input to workspace vector
    t0 = high_resolution_clock::now(); // start

    for (size_t t = 0; t < length; t++)
    {
        for (size_t y = 0; y < height; y++)
        {
            copy(img_seq.data() + t * (height * width) + y * width,
                 img_seq.data() + t * (height * width) + (y + 1) * width,
                 workspace1->begin() + t * (2 * _nx * ny) + y * 2 * _nx);
        }
    }

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Input copy (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Execute fft2 plan
    t0 = high_resolution_clock::now(); // start

    fftw_execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) to preserve Parseval theorem
    double norm_fact = sqrt((double)(nx * ny));
    for (size_t ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        (*workspace1)[ii] /= norm_fact;
    }

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "FFT2 execution (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Compute the image structure function
    t0 = high_resolution_clock::now(); // start

    // initialize helper vector used in average part
    vector<double> tmp(bundle_size, 0.0);
    for (size_t i = 0; i < (_nx * ny - 1) / bundle_size + 1; i++)
    {
        // Step1: correlation part
        // copy values to workspace2 for fft
        for (size_t q = 0; q < bundle_size; q++)
        {
            for (size_t t = 0; t < length; t++)
            {
                (*workspace2)[2 * (q * nt + t)] = (*workspace1)[2 * (t * _nx * ny + i * bundle_size + q)];         // real
                (*workspace2)[2 * (q * nt + t) + 1] = (*workspace1)[2 * (t * _nx * ny + i * bundle_size + q) + 1]; // imag
            }
            // set other values to 0
            for (size_t t = length; t < nt; t++)
            {
                (*workspace2)[2 * (q * nt + t)] = 0.0;
                (*workspace2)[2 * (q * nt + t) + 1] = 0.0;
            }
        }

        // compute the fft
        fftw_execute(fft_plan);

        // compute power spectrum of fft
        for (size_t j = 0; j < bundle_size * nt; j++)
        {
            (*workspace2)[2 * j] = (*workspace2)[2 * j] * (*workspace2)[2 * j] + (*workspace2)[2 * j + 1] * (*workspace2)[2 * j + 1]; // real
            (*workspace2)[2 * j + 1] = 0.0; // imag
        }

        // compute ifft
        fftw_execute(fft_plan);

        // Step2: average part
        size_t idx = lags.size() - 1;
        for (size_t t = 0; t < length; t++)
        {
            for (size_t q = 0; q < bundle_size; q++)
            {
                double a = (*workspace1)[2 * (t * _nx * ny + i * bundle_size + q)];     // real
                double b = (*workspace1)[2 * (t * _nx * ny + i * bundle_size + q) + 1]; // imag
                tmp[q] += a * a + b * b;
                a = (*workspace1)[2 * ((length - t - 1) * _nx * ny + i * bundle_size + q)];     // real
                b = (*workspace1)[2 * ((length - t - 1) * _nx * ny + i * bundle_size + q) + 1]; // imag
                tmp[q] += a * a + b * b;
            }

            // add contribution only if delay in list
            if (length - t - 1 == lags[idx])
            {
                for (size_t q = 0; q < bundle_size; ++q)
                {
                    // also divide corr part by nt to normalize fft
                    (*workspace2)[2 * (q * nt + (size_t)(lags[idx]))] = tmp[q] - 2 * (*workspace2)[2 * (q * nt + (size_t)(lags[idx]))] / (double)nt;
                    // finally, normalize output
                    (*workspace2)[2 * (q * nt + (size_t)(lags[idx]))] /= (double)(length - lags[idx]);
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
        for (size_t idx = 0; idx < lags.size(); idx++)
        {
            for (size_t q = 0; q < bundle_size; q++)
            {
                (*workspace1)[2 * (idx * _nx * ny + i * bundle_size + q)] = (*workspace2)[2 * (q * nt + (size_t)(lags[idx]))];
            }
        }
    }

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "ISF calculation (s):\n" + to_string(dt.count()) + "\n\n";

    /*
    // Make full image structure function (keep real part and copy symmetric part)
    t0 = high_resolution_clock::now(); // start

    make_full_isf(*workspace1,
                  nx,
                  ny,
                  lags.size());

    t1 = high_resolution_clock::now();  // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    logs += "Conversion to full ISF (s):" + endl + to_string(dt.count()) + endl + endl;

    // FFTshift before output
    t0 = high_resolution_clock::now(); // start

    fft2_shift(*workspace1,
               nx,
               ny,
               lags.size());

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    logs += "FFTshift ISF (s):\n" + to_string(dt.count()) + "\n\n";
     */

    // Convert raw output to full and shifted ISF
    t0 = high_resolution_clock::now(); // start

    make_full_shifted_isf(*workspace1,
                          nx,
                          ny,
                          lags.size());

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Conversion to full ISF (s):\n" + to_string(dt.count()) + "\n\n";

    // ***Shrink workspace if needed
    t0 = high_resolution_clock::now(); // start

    // the full size of the image structure function is
    // nx * ny * #(lags)
    workspace1->resize(nx * ny * lags.size());
    workspace1->shrink_to_fit();

    // Cleanup before finish
    fftw_destroy_plan(fft2_plan);
    fftw_destroy_plan(fft_plan);
    fftw_cleanup();
    workspace2->clear();
    workspace2->shrink_to_fit();
    tmp.clear();
    tmp.shrink_to_fit();

    t1 = high_resolution_clock::now(); // stop
    dt = duration_cast<duration<double>>(t1 - t0);
    _logger += "Garbage cleaning (s):\n" + to_string(dt.count()) + "\n\n";

    _logger.clear();

    // Return result to python
    return vector2numpy(workspace1, nx, ny, lags.size());
}

/*!
    Export dfm functions to python.
 */
void export_dfm(py::module &m)
{
    m.def("dfm_direct", &dfm_direct<double>);
    m.def("dfm_direct", &dfm_direct<float>);
    m.def("dfm_direct", &dfm_direct<int64_t>);
    m.def("dfm_direct", &dfm_direct<int32_t>);
    m.def("dfm_direct", &dfm_direct<int16_t>);
    m.def("dfm_direct", &dfm_direct<uint64_t>);
    m.def("dfm_direct", &dfm_direct<uint32_t>);
    m.def("dfm_direct", &dfm_direct<uint16_t>);
    m.def("dfm_direct", &dfm_direct<uint8_t>);
    m.def("dfm_fft", &dfm_fft<double>);
    m.def("dfm_fft", &dfm_fft<float>);
    m.def("dfm_fft", &dfm_fft<int64_t>);
    m.def("dfm_fft", &dfm_fft<int32_t>);
    m.def("dfm_fft", &dfm_fft<int16_t>);
    m.def("dfm_fft", &dfm_fft<uint64_t>);
    m.def("dfm_fft", &dfm_fft<uint32_t>);
    m.def("dfm_fft", &dfm_fft<uint16_t>);
    m.def("dfm_fft", &dfm_fft<uint8_t>);
}