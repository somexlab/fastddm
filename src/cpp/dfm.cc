// Maintainer: enrico-lattuada

/*! \file dfm.cc
    \brief Definition of core C++ Digital Fourier Microscopy functions
*/

// *** headers ***
#include "dfm.h"
#include "pycomm.h"

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

    // Copy input to workspace vector
    auto buff = img_seq.request();  // get pointer to values
    size_t length = buff.shape[0];  // get length of original input
    size_t height = buff.shape[1];  // get height of original input
    size_t width = buff.shape[2];   // get width of original input

    for (size_t t = 0; t < length; t++)
    {
        for (size_t y = 0; y < height; y++)
        {
            copy(img_seq.data() + t * (height * width) + y * width,
                 img_seq.data() + t * (height * width) + (y + 1) * width,
                 workspace->begin() + t * (2 * (nx / 2 + 1) * ny) + y * 2 * (nx / 2 + 1));
        }
    }

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