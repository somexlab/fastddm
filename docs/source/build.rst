.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

.. _build:

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

- Python >= 3.9
- Pip
- C++14 capable compiler (tested with ``gcc`` [Ubuntu], ``clang`` [MacOS], and ``msvc`` [Windows])

Make sure that Python is available from your ``PATH``.
Follow the instructions below to obtain the compiler for your system.

.. tabs::

   .. group-tab:: Ubuntu

      ``gcc`` can be installed by running from the terminal:

      .. code-block:: bash

          $ sudo apt update      
          $ sudo apt install build-essentials

      If you are using other Linux distros, look for the appropriate package and package manager.
      Ensure that the compiler was succesfully installed by running:

      .. code-block:: bash

          $ g++ --version

   .. group-tab:: Mac OSX

      ``clang`` can be installed on macOS by running from the terminal:

      .. code-block:: bash

          $ xcode-select --install

      Ensure that the compiler was succesfully installed by running:

      .. code-block:: bash

          $ clang --version

   .. group-tab:: Windows

      On Windows, you can obtain a C++ compiler by installing Visual Studio Community Edition and
      enabling the ``Desktop development with C++`` option.

      To ensure that the compiler was succesfully installed, open the Developer Command Prompt for VS
      and execute:

      .. code-block:: shell

          > cl

.. warning::
     
   If the installation fails, delete the `build` and `fastddm.egg-info` directories created by the
   installer before making a new attempt!
   This ensures that CMake generates fresh makefiles.

**For GPU execution** (required when ``ENABLE_CUDA=ON``):

- `NVIDIA CUDA Toolkit`_ >= 9.0

First and foremost, you will need a CUDA-capable GPU.
Check your hardware compatibility on the `NVIDIA website <https://developer.nvidia.com/cuda-gpus>`_.
Follow the `instructions <https://docs.nvidia.com/cuda/>`_ available from the website specific to your OS.

.. warning::

   Check the compatibility of your compiler with the CUDA Toolkit version you are installing.

.. note::

   The CUDA Toolkit is not available for MacOS.
   If you are using a Mac, you can skip this step.

**To build the documentation:**

- sphinx
- nbsphinx
- sphinx-tabs
- ipython
- matplotlib
- furo

**To run the unit-tests:**

- pytest
- pytest-regtest

.. _virtual environment: https://docs.python.org/3/library/venv.html
.. _NVIDIA CUDA Toolkit: https://developer.nvidia.com/cuda-downloads

.. _Obtain the source:

Obtain the source
-----------------

Clone using Git_::

  $ git clone https://github.com/somexlab/fastddm.git

Release tarballs are also available as `GitHub release`_ assets.

.. _GitHub release: https://github.com/somexlab/fastddm/releases
.. _Git: https://git-scm.com/


.. _Build and install the package:

Build and install the package
-----------------------------

To build and install from source, run the following command in a terminal from within the
source directory:

.. code-block:: bash

    $ pip3 install .

To install also the dependencies for ``test``, run this command instead:

.. code-block:: bash

    $ pip3 install .[test]

.. warning::
     
   In some cases, notably on Windows and using Z shell on other systems as well, you need to run
   alternatively::

     $ pip3 install ."[test]"

To install the optional dependencies to build the documentation, use the option ``doc``.
If you want to install both, separate the options using a comma, for example:

.. code-block:: bash

    $ pip3 install .[test,doc]

.. _Configure:

Configure
---------

**FastDDM**'s cmake configuration accepts a number of options that you can use to customize your
installation from source:

- ``ENABLE_CPP`` - When enabled, build the core C++ library (default: ``OFF``).
- ``ENABLE_CUDA`` - When enabled, build the core CUDA library (default: ``OFF``). If ``ON``, ``ENABLE_CPP`` is also enabled automatically.
- ``SINGLE_PRECISION`` - Enable single precision output (default: ``OFF``).

.. note::

   ``ENABLE_CUDA`` is available for Linux and Windows only.

.. tip::

   ``SINGLE_PRECISION`` can give advantages on laptops or systems with small RAM size.

Options can be set by giving extra arguments to the ``pip`` command.
To enable any of the options, add ``--config-settings=cmake.define.<OPTION>=<VALUE>``.
For example, to enable the C++ core and single precision math, run:

.. code-block:: bash

    $ pip3 install . --config-settings=cmake.define.ENABLE_CPP=ON --config-settings=cmake.define.SINGLE_PRECISION=ON

To enable the CUDA core, run:

.. code-block:: bash

    $ pip3 install . --config-settings=cmake.define.ENABLE_CUDA=ON

.. tip::

   To speed up the installation, we recommend to use ``uv``. Install it with:

   .. code-block:: bash

      $ pip3 install uv

   Then, use ``uv`` instead of ``pip3`` to install the package:

   .. code-block:: bash

      $ uv pip install .


Using the install script
------------------------

FastDDM provides a convenience script, ``install.py``, to simplify installation and configuration. This script wraps the standard pip (or uv) install process and allows you to easily enable advanced features, extras, and CMake options without lengthy command lines.

.. note::

   The install script is recommended for most users, especially when enabling advanced features or extras.

To enable the C++ core with the install script, the command is:


.. tabs::

   .. group-tab:: Ubuntu

      .. code-block:: bash

         $ python3 install.py --cpp

   .. group-tab:: Mac OSX

      .. code-block:: bash

         $ python3 install.py --cpp

   .. group-tab:: Windows

      Basic install:

      .. code-block:: shell

         > python install.py --cpp

.. tip::

   To see all available and up-to-date options, run:

   .. code-block:: bash

      $ python3 install.py --help

   (or ``python install.py --help`` on Windows).


.. _Test the package:

Test the package
----------------

To test the installation, start python and try importing the package:

.. code-block:: python

    import fastddm

    fastddm.__version__

To run the unit-tests, run the following command from within the source directory:

.. code-block:: bash

    $ pytest -v

.. warning::

   If you want to run the tests, you need to install the test dependencies.

.. _Build the documentation:

Build the documentation
-----------------------

To build the documentation, run the following command from within the source directory:

.. code-block:: bash

    $ sphinx-build -b html docs/source/ docs/build/html
