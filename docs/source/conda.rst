.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

.. _conda:

Build in a conda environment
============================

Creating a dedicated Conda environment is a best practice that ensures the dependencies are managed
effectively.
This isolation prevents conflicts between packages and allows for a clean workspace.
In the following section, we’ll guide you through the process of setting up a Conda environment and
installing FastDDM to get your project up and running smoothly.

1. `Install miniconda`_

2. `Create an environment config YAML file`_

3. `Notes on CUDA`_

.. _Install miniconda:

Install miniconda
-----------------

Download and install miniconda3 from the
`Anaconda website <https://docs.anaconda.com/free/miniconda/index.html>`_.

.. _Create an environment config YAML file:

Create an environment config YAML file
--------------------------------------

Create a ``fastddm-env.yml`` file and write the following content in it (select your operating
system).

.. tabs::

   .. group-tab:: Ubuntu

      .. code-block:: yaml

          name: fddm-env
          channels:
            - defaults
          dependencies:
            - gcc
            - g++
            - python>=3.8
            - pip

      Create the environment by running the following command in your terminal

      .. code-block:: bash

          $ conda env create -f fastddm-env.yml 

      Activate the environment

      .. code-block:: bash

          $ conda activate fddm-env

      Export the environment variables

      .. code-block:: bash

          $ conda env config vars set CC=$CONDA_PREFIX/bin/gcc
          $ conda env config vars set CXX=$CONDA_PREFIX/bin/g++

      To compile the C++ core, also set the corresponding flag

      .. code-block:: bash

          $ conda env config vars set ENABLE_CPP=ON

      Deactivate and reactivate the environment to make the changes effective

      .. code-block:: bash

          $ conda deactivate
          $ conda activate fddm-env

      From the ``fastddm`` project root directory (see :ref:`build` on how to get the source code),
      install the package and the test dependencies

      .. code-block:: bash

          $ pip3 install ."[test]"

      Finally, run the tests from the project source directory

      .. code-block:: bash

          $ pytest -v

   .. group-tab:: MacOS

      .. code-block:: yaml

          name: fddm-env
          channels:
            - defaults
          dependencies:
            - clang
            - clangxx
            - python>=3.8
            - pip

      Create the environment by running the following command in your terminal

      .. code-block:: bash

          $ conda env create -f fastddm-env.yml 

      Activate the environment

      .. code-block:: bash

          $ conda activate fddm-env

      Export the environment variables

      .. code-block:: bash

          $ conda env config vars set CC=$CONDA_PREFIX/bin/clang
          $ conda env config vars set CXX=$CONDA_PREFIX/bin/clang++

      To compile the C++ core, also set the corresponding flag

      .. code-block:: bash

          $ conda env config vars set ENABLE_CPP=ON

      Deactivate and reactivate the environment to make the changes effective

      .. code-block:: bash

          $ conda deactivate
          $ conda activate fddm-env

      From the ``fastddm`` project root directory (see :ref:`build` on how to get the source code),
      install the package and the test dependencies

      .. code-block:: bash

          $ pip3 install ."[test]"

      Finally, run the tests from the project source directory

      .. code-block:: bash

          $ pytest -v

   .. group-tab:: Windows

      .. code-block:: yaml

          name: fddm-env
          channels:
            - defaults
          dependencies:
            - python>=3.8
            - pip

      For Windows, you will still need to install Visual Studio Community Edition with the
      ``Desktop development with C++`` option enabled (see :ref:`build`).
      
      Create the environment by running the following command in your miniconda PowerShell terminal

      .. code-block:: bash

          $ conda env create -f fastddm-env.yml 

      Activate the environment

      .. code-block:: bash

          $ conda activate fddm-env

      To compile the C++ core, set the corresponding flag

      .. code-block:: bash

          $ conda env config vars set ENABLE_CPP=ON

      Deactivate and reactivate the environment to make the changes effective

      .. code-block:: bash

          $ conda deactivate
          $ conda activate fddm-env

      From the ``fastddm`` project root directory (see :ref:`build` on how to get the source code),
      install the package and the test dependencies

      .. code-block:: bash

          $ pip3 install ."[test]"

      Finally, run the tests from the project source directory

      .. code-block:: bash

          $ pytest -v

.. _Notes on CUDA:

Notes on CUDA
-------------

As of today, we could not find a way to automatically build the package from source using the
``cudatoolkit-dev`` distributed on ``conda-forge``.
We recommend following the instructions given in :ref:`build` to install the package in the conda
environment using the system CUDA Toolkit.

We welcome contributions on this matter!
