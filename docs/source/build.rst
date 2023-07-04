.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

Building from source
====================



Windows
-------

Tested with anaconda3.

To install python only package, just run from the project root directory

.. code-block:: bash

    python -m pip install .


To install C++ library, export the ``ENABLE_CPP`` environment variable as follows:

.. code-block:: bash

    $Env:ENABLE_CPP = "ON"

and install as before.
