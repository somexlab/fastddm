Contributing
============

Contributions are welcomed via `pull requests on GitHub
<https://github.com/somexlab/fastddm/pulls>`__. Contact the **FastDDM** developers before
starting work to ensure it meshes well with the planned development direction and standards set for
the project.

Features
--------

Implement functionality in a general and flexible fashion
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

New features should be applicable to a variety of use-cases. The **FastDDM** developers can
assist you in designing flexible interfaces.

Optimize for the current GPU generation
"""""""""""""""""""""""""""""""""""""""

Write, test, and optimize your GPU kernels on the latest generation of GPUs.

Version control
---------------

Guidelines for version control here...

Propose a minimal set of related changes
""""""""""""""""""""""""""""""""""""""""

All changes in a pull request should be closely related. Multiple change sets that are loosely
coupled should be proposed in separate pull requests.

Agree to the Contributor Agreement
""""""""""""""""""""""""""""""""""

All contributors must agree to the Contributor Agreement before their pull request can be merged.
WE NEED A CONTRIBUTOR AGREEMENT!!!

Source code
-----------

Use a consistent style
""""""""""""""""""""""

The **Code style** section of the documentation sets the style guidelines for **FastDDM** code.

Document code with comments
"""""""""""""""""""""""""""

Use doxygen header comments for classes, functions, etc. Also comment complex sections of code so
that other developers can understand them.

Compile without warnings
""""""""""""""""""""""""

Your changes should compile without warnings.

Tests
-----

Write unit tests
""""""""""""""""

Add unit tests for all new functionality.

Validity tests
""""""""""""""

The developer should run research-scale tests using the new functionality and ensure that it
behaves as intended.

User documentation
------------------

Write user documentation
""""""""""""""""""""""""

Document public-facing API with Python docstrings in Google style.

Document version status
"""""""""""""""""""""""

Add `versionadded, versionchanged, and deprecated Sphinx directives
<https://www.sphinx-doc.org/en/master/usage/restructuredtext/directives.html#directive-versionadded>`__
to each user-facing Python class, method, etc., so that users will be aware of how functionality
changes from version to version. Remove this when breaking APIs in major releases.

Add developer to the credits
""""""""""""""""""""""""""""

Update the credits documentation to list the name and affiliation of each individual that has
contributed to the code.

Propose a change log entry
""""""""""""""""""""""""""

Propose a short concise entry describing the change in the pull request description.
