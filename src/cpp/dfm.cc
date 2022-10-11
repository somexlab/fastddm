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
                               size_t ny,
                               size_t nt)
{
    // Allocate workspace vector
    /*
    - Vector needs to be allocated on heap so that on return
      `vector2numpy` can take ownership
    - We need to make sure that the fft2 r2c fits in the array,
      so the size of one fft2 output is ny*(nx//2 + 1) complex
      doubles [the input needs to be twice as large]
     */
    vector<double> *workspace = new vector<double>(2 * (nx / 2 + 1) * ny * nt, 0.0);

    // Create the fft2 plan
    fftw_plan fft2_plan = fft2_create_plan(*workspace,
                                           nx,
                                           ny,
                                           nt);

    // Copy input to workspace vector
    auto buff = img_seq.request(); // get pointer to values
    size_t length = buff.shape[0]; // get length of original input
    size_t height = buff.shape[1]; // get height of original input
    size_t width = buff.shape[2];  // get width of original input

    for (size_t t = 0; t < length; t++)
    {
        for (size_t y = 0; y < height; y++)
        {
            copy(img_seq.data() + t * (height * width) + y * width,
                 img_seq.data() + t * (height * width) + (y + 1) * width,
                 workspace->begin() + t * (2 * (nx / 2 + 1) * ny) + y * 2 * (nx / 2 + 1));
        }
    }

    // Execute fft2 plan
    fftw_execute(fft2_plan);

    // Normalize fft2
    // Use sqrt(num_pixels) to preserve parseval theorem
    double norm_fact = sqrt((double)(nx * ny));
    for (size_t ii = 0; ii < 2 * (nx / 2 + 1) * ny * nt; ii++)
    {
        (*workspace)[ii] /= norm_fact;
    }

    // Compute the image structure function
    // initialize helper vector
    vector<double> tmp(lags.size(), 0.0);

    // loop over the q values
    for (size_t q = 0; q < nx * ny; q++)
    {
        // zero out the helper vector
        fill(tmp.begin(), tmp.end(), 0);

        // loop over the lags
        for (size_t _dt = 0; _dt < lags.size(); _dt++)
        {
            // get current lag
            size_t dt = lags[_dt];

            // loop over time
            for (size_t t = 0; t < nt - dt; t++)
            {
                // compute the power spectrum of the difference of pixel at time t and time t+dt, i.e.
                // [(a+ib) - (c+id)] * conj[(a+ib) - (c+id)] = (a-c)^2+(b-d)^2
                // notice fft is complex, so the stride between two consecutive pixels is two
                double a = (*workspace)[2 * ((t + dt) * (nx * ny) + q)];
                double b = (*workspace)[2 * ((t + dt) * (nx * ny) + q) + 1];
                double c = (*workspace)[2 * ((t) * (nx * ny) + q)];
                double d = (*workspace)[2 * ((t) * (nx * ny) + q) + 1];

                tmp[_dt] += (a - c) * (a - c) + (b - d) * (b - d);
            }

            // normalize
            tmp[_dt] /= (double)(nt - dt);
        }

        // copy the values back in the vector
        copy_vec_with_stride(tmp,
                             *workspace,
                             2 * q,
                             2 * (nx * ny));
    }

    // keep real part only (stride vector from 2 to 1)
    complex2real(*workspace,
                 nx * ny * lags.size());

    // Cleanup before finish
    fftw_destroy_plan(fft2_plan);
    fftw_cleanup();
    tmp.clear();
    tmp.shrink_to_fit();

    // Return result to python
    return vector2numpy(workspace, 2 * (nx / 2 + 1), ny, nt);
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
}