Contributing
============

Contributions are welcomed via `pull requests on GitHub
<https://github.com/somexlab/fastddm/pulls>`__. Contact the **FastDDM** developers before
starting work to ensure it meshes well with the planned development direction and standards set for
the project. (Contact Roberto Cerbino via email at roberto.cerbino@univie.ac.at.)

Features
--------

Request features or report bugs
"""""""""""""""""""""""""""""""

Feature requests or bug reports should be done through
`issues <https://github.com/somexlab/fastddm/issues>`__.
Please, create a new issue following the provided template.
Also consider adding proper labels, like ``bug`` or ``enhancement`` / ``task``.
Please, be concise in your description: adding screenshots (or mathematical formulas) may also help!
The maintainers will then add the assignee and appropriate milestone.
Or, if you want to contribute yourself, let the **FastDDM** developers know
and **read carefully below**.

Implement functionality in a general and flexible fashion
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

New features should be applicable to a variety of use-cases. The **FastDDM** developers can
assist you in designing flexible interfaces.

Pull requests
"""""""""""""

When opening a pull request, make sure that the source and destination branches are compatible,
e.g. ``<your_branch> -> <latest_milestone_branch>`` or ``<latest_milestone_branch> -> main``.

If you want to contribute to the project, fork the latest milestone or the main branch.
When you are done with your modifications, open a pull request.

If you are an internal developer, while you're working on the PR,
please use the title format "WIP: \<title>". Once you've finished your work,
remove the WIP part and ask for a review from one of the repository's maintainers.

If you need advice, add comments or open a discussion, possibly tagging the appropriate maintainer.

Add your Python file to the installer
"""""""""""""""""""""""""""""""""""""

Python modules must be added to the ``src/python`` directory. To make a module installable,
add it to the ``src/CMakeLists.txt`` file.
Scroll down until you find

.. code:: cmake
    
    set(python_SOURCES
        python/__init__.py
        ...
    )

Add your file to the list.

Version control
---------------

When working on the issue, create a new branch starting from the latest
milestone branch or from ``main``.

Pull requests for new releases
""""""""""""""""""""""""""""""

After merging the latest release into ``main``, open a new WIP release PR. Do so by:

#. creating a new branch ``vX.Y.Z`` following the convention.
#. updating the fallback version in ``setup.py`` and in ``src/python/make_install_setup.py``.
#. updating the version in ``docs/source/conf.py``
#. committing to the new branch.
#. opening a PR with title "WIP: vX.Y.Z" with destination branch ``main``.

Merge release into `main`
"""""""""""""""""""""""""

#. update changelog with the main modifications of the release you want to merge.
   Follow the convention in the ``CHANGELOG.md``.
#. Merge the PR to ``main`` & close the corresponding milestone.
#. Create a tag via "create draft release" and name it according to the release version.

Propose a minimal set of related changes
""""""""""""""""""""""""""""""""""""""""

All changes in a pull request should be closely related. Multiple change sets that are loosely
coupled should be proposed in separate pull requests.

Agree to the Contributor Agreement
""""""""""""""""""""""""""""""""""

All contributors must agree to the Contributor Agreement before their pull request can be merged.
Send an email to roberto.cerbino@univie.ac.at to confirm.

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

Document public-facing API with Python docstrings in NumPy style.

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
