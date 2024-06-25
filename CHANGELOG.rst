
Change Log
==========

v0.3
----

v0.3.11 (2024-06-25)
^^^^^^^^^^^^^^^^^^^^

*Added*

* Added to documentation an explanation of the structure of binary files saved by the library

*Changed*

* Updated pybind11 version to 2.12.0 (to support NumPy 2.x)
* Renamed the ``_utils`` module to ``utils`` to make it publicly accessible

*Fixed*

* Fixed links and other small issues in the documentation

.. *Deprecated*
.. *Removed*

v0.3.10 (2024-04-04)
^^^^^^^^^^^^^^^^^^^^

.. *Added*
.. *Changed*

*Fixed*

* Documentation build errors

.. *Deprecated*
.. *Removed*

v0.3.9 (2024-04-04)
^^^^^^^^^^^^^^^^^^^

*Added*

* Instructions for installation via PyPI and on Google Colab

*Changed*

* Updated minimal example notebook

*Fixed*

* Windows' PyPI wheels

.. *Deprecated*
.. *Removed*

v0.3.8 (2024-03-28)
^^^^^^^^^^^^^^^^^^^

*Added*

* PyPI badge to README
* Precompiled wheels to PyPI

.. *Changed*
.. *Fixed*
.. *Deprecated*
.. *Removed*

v0.3.7 (2024-03-28)
^^^^^^^^^^^^^^^^^^^

.. *Added*

.. *Changed*

*Fixed*

* Changed wheel creation trigger behavior

.. *Deprecated*
.. *Removed*

v0.3.6 (2024-03-28)
^^^^^^^^^^^^^^^^^^^

.. *Added*

.. *Changed*

*Fixed*

* Explicitly set author-email in pyproject.toml to avoid display issues on PyPI

.. *Deprecated*
.. *Removed*

v0.3.5 (2024-03-28)
^^^^^^^^^^^^^^^^^^^

.. *Added*

.. *Changed*

*Fixed*

* Due to PyPI maintainance, ``v0.3.4`` was not published to the package index. It should happen now.

.. *Deprecated*
.. *Removed*

v0.3.4 (2024-03-28)
^^^^^^^^^^^^^^^^^^^

*Added*

* Source distribution released to PyPI
* Add instructions to build sdist to documentation

.. *Changed*
.. *Fixed*
.. *Deprecated*
.. *Removed*

v0.3.3 (2024-03-28)
^^^^^^^^^^^^^^^^^^^

*Added*

* Optional dependencies can be used during installation
* Instructions for Conda environment
* Prepare project for release to PyPI

*Changed*

* Code style for C++/CUDA source
* Build instructions have been updated

*Fixed*

* Installation in conda environments and on systems with multiple compilers and Python versions
* Small correction to example in docs landing page

.. *Deprecated*
.. *Removed*

v0.3.2 (2024-03-07)
^^^^^^^^^^^^^^^^^^^

*Added*

* Build and test badge to documentation
* Azimuthal average test

.. *Changed*

*Fixed*

* Resolved bug where list input in ``azimuthal_average`` was throwing an error

.. *Deprecated*
.. *Removed*

v0.3.1 (2024-03-05)
^^^^^^^^^^^^^^^^^^^

*Added*

* Documentation is now hosted on ReadtheDocs
* Build and tests (currently, only Python and C++; CUDA not possible on GitHub actions)

*Changed*

* README file
* Updates to developers documentation

.. *Fixed*
.. *Deprecated*
.. *Removed*

v0.3.0 (2024-03-01)
^^^^^^^^^^^^^^^^^^^

*Added*

* Image windowing function can now be used as an input for `ddm` function.
* Documentation (developer guidelines for contributors, installation, doc-strings, etc.)
* Noise estimators 
* Intermediate scattering function
* Logo
* support for fixed expression in `fit_multik`
* general function `azimuthal_average_array`  to perform an azimuthal average on a 3D `ndarray`

*Changed*

* `fit_multik` also returns the standard error of the parameters
* refactored functions for C++ & CUDA
* improved azimuthal average (changed behaviour of input bins variable)

*Fixed* 

* mergesort & resample bugs with different dtypes
* initial fit at reference k value in `fit_multik` was not done properly
* CUDA compilation works well on various systems

.. *Deprecated*
.. *Removed*


v0.2
----

v0.2.0 (2023-04-20)
^^^^^^^^^^^^^^^^^^^

*Added*

* ``ImageStructureFunction`` provides methods to retrieve the full plane representation from half-plane.
* ``SINGLE_PRECISION`` option at install time.
* Errors evaluated in ``AzimuthalAverage`` can be used in ``fit_multik``
* In ``fit_multik``, the user can now fix q-dependent parameter values via ``fixed_params``.
* In ``fit_multik``, the user can now fix q-dependent parameters range via ``fixed_params_min`` and ``fixed_params_max``.
* Results from ``fit_multik`` also include the ``k`` parameter for convenience.

*Changed*

* Now ``ImageStructureFunction`` data and err are stored using half-plane representation.
* Fit models are not saved to file due to incompatibilities with ``dill`` package.
* Updated pytest.

*Fixed* 

* Passed parameters in ``fit_multik`` are not changed by the function.

.. *Deprecated*

*Removed*

* Removed unused functions and modules.

v0.1
----

v0.1.3 (2023-04-04)
^^^^^^^^^^^^^^^^^^^

*Added*

* Reader function for raw image metadata (supports ``.tif`` and ``.nd2``)
* basic pytest routines
* ``chunkify`` function for time analysis (for non-stationary processes)
* variance in python backend
* uncertainty in azimuthal average (selectable via flag in ``azimuthal_average`` function)
* fit function to fit a model for multiple k/q values at once
* copyright information
* script to generate fit models for the intermediate scattering function and image structure function
* formatted (custom) binary file outputs (and readers/parsers) for the ``ImageStructureFunction`` and the ``AzimuthalAverage`` classes


*Changed*

* Now ``save`` method in ``ImageStructureFunction`` and ``AzimuthalAverage`` allow the usage of the filename without the argument keyword.
* ``read_images`` and related functions allow for the selection of color channels when reading image files.
* Improved speed in cuda fft2 step.
* moved from static to dynamic library compilation 
* sped up the azimuthal average computation 

*Fixed* 

* fixed a bug in azimuthal average ``resample`` method
* fixed scaling issue in python backend fft mode
* fixed bug in variance calculation in cuda backend
* computing lag=0 in ddm python backend was causing an error due to faulty array slicing, this was fixed. 

.. *Deprecated*
.. *Removed*

v0.1.2 (2023-01-24)
^^^^^^^^^^^^^^^^^^^

*Added*

* New ``power_spec`` and ``var`` properties in ``ImageStructureFunction`` and ``AzimuthalAverage`` classes, containing the average power spectrum of the input images and the variance of the temporal fluctuations of the 2D Fourier transformed images.
* ``len()`` method in ``ImageStructureFunction`` class, which returns the length of the underlying ``data``.

.. *Changed*

*Fixed*

* Fixed memory leak.
* Fixed evaluation of RAM memory in ``cuda`` core functions.

.. *Deprecated*
.. *Removed*

v0.1.1 (2023-01-20)
^^^^^^^^^^^^^^^^^^^

*Added*

* New ``shape`` property in ``AzimuthalAverage`` class.
* Unified ``read_images`` function with support for Nikon .nd2 files.

*Changed*

* ``azimuthal_average`` method now works only with ``ImageStructureFunction`` objects. The generic method was moved to ``_azimuthal_average``.
* ``pixel_size`` and ``delta_t`` properties of ``ImageStructureFunction`` class can now be set by simple assignment and work as previous ``set_pixel_size`` and ``set_delta_t`` methods.
* Improved performance of ``azimuthal_average`` when ``weights`` are not set.
* Improved speed of optimization step in cuda functions.

*Fixed*

* Fixed bug in image sequence reader.
* Fixed fatal error on import in macOS systems when using conda environment.

.. *Deprecated*

*Removed*

* ``set_pixel_size`` and ``set_delta_t`` methods of ``ImageStructureFunction`` class have been substituted with simple assignment.
* Removed unused dependencies.

v0.1.0 (2023-01-16)
^^^^^^^^^^^^^^^^^^^

*Added*

* Image reading functions.
* Image structure function calculation on CPU (python and C++) and GPU.
* Azimuthal average calculation on CPU (python only).
* Basic windowing functions for image preprocessing.
* Basic central cross mask for azimuthal average computation.
* Basic weights (sector average) functions for azimuthal average.
* Objects loading and saving.
* Lag selection functions.
* Fit interface with ``lmfit``.

.. *Changed*
.. *Fixed*
.. *Deprecated*
.. *Removed*
