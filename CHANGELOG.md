
Change Log
==========

v0.2
----

### v0.2.0 (2023-04-20)

*Added*

* `ImageStructureFunction` provides methods to retrieve the full plane representation from half-plane.
* `SINGLE_PRECISION` option at install time.
* Errors evaluated in `AzimuthalAverage` can be used in `fit_multik`
* In `fit_multik`, the user can now fix q-dependent parameter values via `fixed_params`.
* In `fit_multik`, the user can now fix q-dependent parameters range via `fixed_params_min` and `fixed_params_max`.
* Results from `fit_multik` also include the `k` parameter for convenience.

*Changed*

* Now `ImageStructureFunction` data and err are stored using half-plane representation.
* Fit models are not saved to file due to incompatibilities with `dill` package.
* Updated pytest.

*Fixed* 

* Passed parameters in `fit_multik` are not changed by the function.

[comment]: <> (*Deprecated*)

*Removed*

* Removed unused functions and modules.

v0.1
----

### v0.1.3 (2023-04-04)

*Added*

* Reader function for raw image metadata (supports `.tif` and `.nd2`)
* basic pytest routines
* `chunkify` function for time analysis (for non-stationary processes)
* variance in python backend
* uncertainty in azimuthal average (selectable via flag in `azimuthal_average` function)
* fit function to fit a model for multiple k/q values at once
* copyright information
* script to generate fit models for the intermediate scattering function and image structure function
* formatted (custom) binary file outputs (and readers/parsers) for the `ImageStructureFunction` and the `AzimuthalAverage` classes


*Changed*

* Now `save` method in `ImageStructureFunction` and `AzimuthalAverage` allow the usage of the filename without the argument keyword.
* `read_images` and related functions allow for the selection of color channels when reading image files.
* Improved speed in cuda fft2 step.
* moved from static to dynamic library compilation 
* sped up the azimuthal average computation 

*Fixed* 

* fixed a bug in azimuthal average `resample` method
* fixed scaling issue in python backend fft mode
* fixed bug in variance calculation in cuda backend
* computing lag=0 in ddm python backend was causing an error due to faulty array slicing, this was fixed. 

[comment]: <> (*Deprecated*)

[comment]: <> (*Removed*)

### v0.1.2 (2023-01-24)

*Added*

* New `power_spec` and `var` properties in `ImageStructureFunction` and `AzimuthalAverage` classes, containing the average power spectrum of the input images and the variance of the temporal fluctuations of the 2D Fourier transformed images.
* `len()` method in `ImageStructureFunction` class, which returns the length of the underlying `data`.

[comment]: <> (*Changed*)

*Fixed*

* Fixed memory leak.
* Fixed evaluation of RAM memory in `cuda` core functions.

[comment]: <> (*Deprecated*)

[comment]: <> (*Removed*)

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

[comment]: <> (*Deprecated*)

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

[comment]: <> (*Changed*)

[comment]: <> (*Fixed*)

[comment]: <> (*Deprecated*)

[comment]: <> (*Removed*)
