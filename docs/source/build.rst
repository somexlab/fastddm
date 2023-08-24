.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

Building from source
====================



Windows
-------

We tested the installation of the code on Windows (version >= 10).
Even for the python version only, you still need a working C++ compiler.
We tested the following setup:

- Visual Studio Community edition. This is probably an overkill, but it sets all the
  environment variables.
- To install the CUDA core, you need the
  `NVIDIA CUDA Toolkit <https://developer.nvidia.com/cuda-downloads>`_.
  Follow the instructions on the official
  `documentation <https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html>`_.
- As for Python, we suggest installing `anaconda <https://www.anaconda.com/download>`_.

To install python only package, just run from the project root directory

.. code-block:: bash

    python -m pip install .


To install C++ library, export the ``ENABLE_CPP`` environment variable as follows:

.. code-block:: bash

    $Env:ENABLE_CPP = "ON"

and install as before.


.. _win_cuda_docs: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html
