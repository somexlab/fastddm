.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

Features
========

Differential Dynamic Microscopy
-------------------------------

FastDDM can perform the analysis of image sequences using Differential Dynamic Microscopy.
It provides means to load images on disk in various formats, namely the ones available from the
`scikit-image <https://scikit-image.org/>`_ package and the Nikon ``nd2`` files.
It efficiently computes then image structure function and the azimuthal average and provides tools
for the fit and analysis.

Python package
--------------

FastDDM is a Python package and is designed to interoperate with other packages in the scientific
Python ecosystem and to be extendable in user scripts. To enable interoperability, all operations
provide access to useful computed quantities as properties in native Python types or numpy arrays
where appropriate.

CPU and GPU devices
-------------------

FastDDM can execute the computation of the image structure function on CPUs or GPUs. Typical
analysis run more efficiently on GPUs for image sequences containing more than a few hundreds
images. The provided binaries support NVIDIA GPUs. All other operations are executed on the CPU
using standar Python libraries.

Autotuned kernel parameters
---------------------------

FastDDM automatically tunes the kernel parameters to improve performance when executing on a GPU
device. It also tunes the computational parameters to optimize the memory consumption (using all
the RAM available) when executing on a CPU (using the C++ core) or on a GPU. The functions provide
the same output regardless of the parameter (within floating point precision).

The optimal parameters can depend on the number and the size of the images in the sequence and they
weakly depend on other parameters such as the bit depth.

Mixed precision
---------------

FastDDM can perform the computation of the image structure function with mixed floating point
precision. If the single precision library is installed, all calculations are performed in double
precision and the data are then transferred from and to the GPU in single precision. In the case of
calculations performed on the CPU, all mathematical operations requiring high precision are
performed in double precision mode.
