.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

Installing in Google Colab
==========================

**FastDDM** can be installed in `Google Colab <https://colab.research.google.com/>`_, also with GPU
support. This allows to test the library without the need to install it locally on workstations.

.. tip::

    To better understand what the different options imply, see :ref:`build` and
    :ref:`install_sdist`.

Open a new notebook.
If you wish to use the GPU provided by the host server, change the ``runtime type``.
To do so, select from the toolbar ``Runtime > Change runtime type`` and select a GPU available from
the ``Hardware accelerator`` list.
As of today (3 April 2024), with the free plan, you should have access to the T4 GPU.
Save and go back to your notebook.

To check that the GPU is working, run the following command in a cell

.. code-block:: bash

    !nvidia-smi

From pip source distribution
----------------------------

To install the source distribution via ``pip``, run the command below

.. code-block:: bash

    !pip3 install --no-binary fastddm fastddm

From GitHub repository source code
----------------------------------

This is the preferred mode if you want to run the unit tests.
Clone the source code from the GitHub repository

.. code-block:: bash

    !git clone --depth 1 --branch <version_name> https://github.com/somexlab/fastddm.git

For example, to clone the version ``v0.3.8``, run

.. code-block:: bash

    !git clone --depth 1 --branch v0.3.8 https://github.com/somexlab/fastddm.git

Change directory to the source code one

.. code-block:: bash

    %cd fastddm

If you want to install the library with C++ support, run the following command in a cell

.. code-block:: bash

    %env ENABLE_CPP=ON

If you want to install the library with GPU (CUDA) support, run the following command in a cell

.. code-block:: bash

    %env ENABLE_CUDA=ON

(This also enables the C++ support.)

.. warning::

    Due to the small amount of available RAM memory, we also warmly recommend to enable the single
    precision calculation option::

      %env SINGLE_PRECISION=ON

Finally, install **FastDDM**

.. code-block:: bash

    !python3 -m pip install .

.. tip::

    To run unit tests, you need some additional libraries. Run::

      !python3 -m pip install pytest pytest-regtest

    Then, run the tests with::

      !pytest -v

Now you can go back to your home directory and work with **FastDDM**!

.. code-block:: bash

    %cd ..
