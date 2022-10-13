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
    size_t _nx = nx / 2 + 1;
    vector<double> *workspace = new vector<double>(2 * _nx * ny * length, 0.0);

    // ***Create the fft2 plan
    fftw_plan fft2_plan = fft2_create_plan(*workspace,
                                           nx,
                                           ny,
                                           length);

    // ***Copy input to workspace vector
    for (size_t t = 0; t < length; t++)
    {
        for (size_t y = 0; y < height; y++)
        {
            copy(img_seq.data() + t * (height * width) + y * width,
                 img_seq.data() + t * (height * width) + (y + 1) * width,
                 workspace->begin() + t * (2 * _nx * ny) + y * 2 * _nx);
        }
    }

    // ***Execute fft2 plan
    fftw_execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) to preserve Parseval theorem
    double norm_fact = sqrt((double)(nx * ny));
    for (size_t ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        (*workspace)[ii] /= norm_fact;
    }

    // ***Compute the image structure function
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

    // Make full image structure function (keep real part and copy symmetric part)
    make_full_isf(*workspace,
                  nx,
                  ny,
                  lags.size());

    // FFTshift before output
    fft2_shift(*workspace,
               nx,
               ny,
               lags.size());

    cout << "After shift" << endl
         << endl;
    display_vector(workspace, nx, ny, lags.size());

    // ***Shrink workspace if needed
    // the full size of the image structure function is
    // nx * ny * #(lags)
    workspace->resize(nx * ny * lags.size());
    workspace->shrink_to_fit();

    // Cleanup before finish
    fftw_destroy_plan(fft2_plan);
    fftw_cleanup();
    tmp.clear();
    tmp.shrink_to_fit();

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
    size_t _nx = nx / 2 + 1;
    vector<double> *workspace1 = new vector<double>(2 * _nx * ny * length, 0.0);
    vector<double> *workspace2 = new vector<double>(bundle_size * nt, 0.0);

    // ***Create the fft2 plan
    fftw_plan fft2_plan = fft2_create_plan(*workspace1,
                                           nx,
                                           ny,
                                           length);

    // ***Create the fft and ifft plans
    fftw_plan fft_plan = fft_create_plan(*workspace2,
                                         nt,
                                         bundle_size);

    fftw_plan ifft_plan = ifft_create_plan(*workspace2,
                                           nt,
                                           bundle_size);

    // ***Copy input to workspace vector
    for (size_t t = 0; t < length; t++)
    {
        for (size_t y = 0; y < height; y++)
        {
            copy(img_seq.data() + t * (height * width) + y * width,
                 img_seq.data() + t * (height * width) + (y + 1) * width,
                 workspace1->begin() + t * (2 * _nx * ny) + y * 2 * _nx);
        }
    }

    // ***Execute fft2 plan
    fftw_execute(fft2_plan);

    // ***Normalize fft2
    // use sqrt(num_pixels) to preserve Parseval theorem
    double norm_fact = sqrt((double)(nx * ny));
    for (size_t ii = 0; ii < 2 * _nx * ny * length; ii++)
    {
        (*workspace1)[ii] /= norm_fact;
    }

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