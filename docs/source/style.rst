.. Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
.. Part of FastDDM, released under the GNU GPL-3.0 License.

Code style
==========

All code in FastDDM follows a consistent style to ensure readability. We
provide VSCode configuration files for linters (see below) so that developers can
validate and format files.

Python
------

Python code in FastDDM should follow `PEP8`_ with the formatting performed by
`black`_.

.. _PEP8: https://www.python.org/dev/peps/pep-0008
.. _black: https://github.com/psf/black

Tools
^^^^^

* Linter: `flake8 <http://flake8.pycqa.org/en/latest/>`_

  * With these plugins:

    * `pep8-naming <https://github.com/PyCQA/pep8-naming>`_
    * `flake8-docstrings <https://gitlab.com/pycqa/flake8-docstrings>`_
    * `flake8-rst-docstrings <https://github.com/peterjc/flake8-rst-docstrings>`_

  * Configure flake8 in your editor to see violations on save.

* Autoformatter: `black <https://github.com/psf/black>`_

Documentation
^^^^^^^^^^^^^

Python code should be documented with docstrings and added to the Sphinx
documentation index in ``docs/``. Docstrings should follow `NumPy style`_
formatting for use in `Napoleon`_.

.. _NumPy Style: https://www.sphinx-doc.org/en/master/usage/extensions/example_numpy.html
.. _Napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html

Non self-explanatory methods should unambiguously document what calculations they perform
using formal mathematical notation and use a consistent set of symbols and across the whole
codebase. FastDDM documentation should follow standard physics and mathematics notation with
consistent use of symbols detailed in `notation`.

When referencing classes, methods, and properties in documentation, use ``name`` to refer to names
in the local scope (class method or property, or classes in the same module). For classes outside
the module, use the fully qualified name (e.g. ``numpy.ndarray`` or
``fastddm.azimuthalaverage.AzimuthalAverage``).

To build the documentation, run from the project root

.. code-block:: bash

  sphinx-build -b html docs/source/ docs/build/html

C++/CUDA
--------

* Style convention:

  * Whitesmith's indentation style.
  * 100 character line width.
  * Indent only with spaces.
  * 4 spaces per indent level.

* Naming conventions:

  * Constants: all upper-case with words separated by underscores
    ``SOME_CONSTANT``
  * Functions: ``snake_case``
  * Class names: ``UpperCamelCase``
  * Methods: ``snake_case``
  * Member variables: ``m_`` prefix followed by lowercase with words
    separated by underscores ``m_member_variable``
  * Namespaces: All lowercase ``somenamespace``

Documentation
^^^^^^^^^^^^^

We are currently not compiling any documentation for the C++/CUDA functions.
We have not made any decision regarding the documentation style.
However, we currently use the Qt style, as indicated by the `Doxygen`_ documentation.

.. _Doxygen: https://www.doxygen.nl/manual/docblocks.html

Other file types
----------------

Use your best judgment and follow existing patterns when styling CMake,
restructured text, markdown, and other files. The following general guidelines
apply:

* 100 character line width.
* 4 spaces per indent level.
* 4 space indent.
