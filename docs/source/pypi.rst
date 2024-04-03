.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

Installing from PyPI
====================

Install precompiled wheels
--------------------------

Precompiled wheels are available through `PyPI <https://pypi.org/project/fastddm/>`_ with C++
support for most OSs and Python versions.
The command below will install **FastDDM**

.. code-block:: bash

    $ pip3 install fastddm

.. warning::

    If wheels are not available for your system, ``pip`` will fall back to the installation of the
    source distribution. Refer to the :ref:`build` section for more information.

.. _install_sdist:

Install source distribution
---------------------------

To install **FastDDM** from the source distribution, check first the section :ref:`build` for the
prerequisites for your system and the installation options.
If you are using Anaconda, check also the corresponding section: :ref:`conda`.
Run the following line in your terminal:

.. code-block:: bash

    $ pip3 install --no-binary fastddm fastddm
