
Change Log
==========

v0.x
----

### v0.1.1 (2023-01-20)

*Added*

* New `shape` property in `AzimuthalAverage` class.
* Unified `read_images` function with support for Nikon .nd2 files.

*Changed*

* `azimuthal_average` method now works only with `ImageStructureFunction` objects. The generic method was moved to `_azimuthal_average`.
* `pixel_size` and `delta_t` properties of `ImageStructureFunction` class can now be set by simple assignment and work as previous `set_pixel_size` and `set_delta_t` methods.
* Improved performance of `azimuthal_average` when `weights` are not set.
* Improved speed of optimization step in cuda functions.

*Fixed*

* Fixed bug in image sequence reader.
* Fixed fatal error on import in macOS systems when using conda environment.

.. *Deprecated*

*Removed*

* `set_pixel_size` and `set_delta_t` methods of `ImageStructureFunction` class have been substituted with simple assignment.
* Removed unused dependencies.

### v0.1.0 (2023-01-16)

*Added*

* Image reading functions.
* Image structure function calculation on CPU (python and C++) and GPU.
* Azimuthal average calculation on CPU (python only).
* Basic windowing functions for image preprocessing.
* Basic central cross mask for azimuthal average computation.
* Basic weights (sector average) functions for azimuthal average.
* Objects loading and saving.
* Lag selection functions.
* Fit interface with `lmfit`.

.. *Changed*

.. *Fixed*

.. *Deprecated*

.. *Removed*
