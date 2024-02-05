.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

Building from source
====================

To build the **FastDDM** Python package from source:

1. `Install prerequisites`_

2. `Obtain the source`_

3. `Configure`_

4. `Build and install the package`_

5. `Test the package`_

To build the documentation from source (optional):

1. `Install prerequisites`_

2. `Build the documentation`_

.. _Install prerequisites:

Install prerequisites
---------------------

**FastDDM** requires a number of tools and libraries to build.
The options ``ENABLE_CPP`` and ``ENABLE_CUDA`` each require additional softwares/libraries when enabled.

.. tip::

    Create a `virtual environment`_, one place where you can install dependencies and
    **FastDDM**::

      $ python3 -m venv fastddm-venv

    You will need to activate your environment before configuring **FastDDM**::

      $ source fastddm-venv/bin/activate

**General requirements:**

- C++14 capable compiler (tested with ``gcc``, ``clang``, and ``msvc``)
- Python >= 3.8
- Pip

Make sure that Python is available from your ``PATH``.

.. tabs::

   .. group-tab:: Linux

      ``gcc`` can be installed by running from the terminal:

      .. code-block:: bash
      
          $ sudo apt install build-essentials

      Ensure that the compiler was succesfully installed by running

      .. code-block:: bash

          $ g++ --version

   .. group-tab:: Mac OSX

      ``clang`` can be installed on macOS by running from the terminal

      .. code-block:: bash

          $ xcode-select --install

      Ensure that the compiler was succesfully installed by running

      .. code-block:: bash

          $ clang --version

   .. group-tab:: Windows

      On Windows, you can obtain a C++ compiler by installing Visual Studio Community Edition and
      enabling the ``Desktop development with C++`` option.

      To ensure that the compiler was succesfully installed, open the Developer Command Prompt for VS
      and execute

      .. code-block:: shell

          > cl

**For GPU execution** (required when ``ENABLE_CUDA=ON``):

- `NVIDIA CUDA Toolkit`_ >= 9.0

Follow the `instructions <https://docs.nvidia.com/cuda/>`_ available from the website specific to your OS.
Notice that CUDA is not available for MacOS.

**To build the documentation:**

- sphinx
- nbsphinx
- sphinx-tabs
- ipython
- matplotlib
- furo

.. _virtual environment: https://docs.python.org/3/library/venv.html
.. _NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _Obtain the source:

Obtain the source
-----------------

Clone using Git_::

  $ git clone https://github.com/somexlab/fastddm

Release tarballs are also available as `GitHub release`_ assets.

.. _GitHub release: https://github.com/somexlab/fastddm/releases
.. _Git: https://git-scm.com/

.. _Configure:

Configure
---------

**FastDDM**'s cmake configuration accepts a number of options.
These must be set before installation.

- ``ENABLE_CPP`` - When enabled, build the core C++ library (default: ``OFF``).
- ``ENABLE_CUDA`` - When enabled, build the core CUDA library (default: ``OFF``).
  If ``ON``, ``ENABLE_CPP`` will be set to ``ON`` automatically.
- ``SINGLE_PRECISION`` - Enable single precision output (default: ``OFF``).

``ENABLE_CUDA`` is available for Linux and Windows only.
``SINGLE_PRECISION`` can give advantages on laptops or systems with small RAM size.

.. tabs::

   .. group-tab:: Linux and Mac OSX

      Options can be set through the terminal by running the following command

      .. code-block:: bash

        $ export <variable>=ON

      For example, to set ``ENABLE_CPP`` use:

      .. code-block:: bash

        $ export ENABLE_CPP=ON

   .. group-tab:: Windows

      Options can be set through the terminal by running the following command

      .. code-block:: shell

          > $env:<variable> = 'ON'

      For example, to set ``ENABLE_CPP`` use:

      .. code-block:: shell

          > $env:ENABLE_CPP = 'ON'

.. _Build and install the package:

Build and install the package
-----------------------------

To build and install from source, run the following command in a terminal from within the
source directory:

.. code-block:: bash

    $ pip3 install .

.. _Test the package:

Test the package
----------------

To test the installation, start python and try importing the package:

.. code-block:: python

    import fastddm
    fastddm.__version__

.. _Build the documentation:

Build the documentation
-----------------------

To build the documentation, run the following command from within the source directory

.. code-block:: bash

    $ sphinx-build -b html docs/source/ docs/build/html
