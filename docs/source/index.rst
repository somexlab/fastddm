.. FastDDM documentation master file, created by
   sphinx-quickstart on Thu Jun  1 17:13:03 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to FastDDM's documentation!
===================================

**FastDDM** is a Python package for the analysis of microscopy image sequences using Differential
Dynamic Microscopy on CPU and GPU. The features implemented are targeted at the experimental soft
matter research community dealing with inert and active/biological samples.

Resources
=========

- `GitHub repository <https://github.com/somexlab/fastddm>`_:
  Source code and issue tracker.
- `Example notebooks <https://github.com/somexlab/fastddm-tutorials>`_:
  Jupyter notebooks with practical examples.

Example scripts
===============

These examples demonstrate some of the Python API.

Calculation of the image structure function and its azimuthal average:

.. code:: python

    import fastddm as fddm

    file_names = [...]  # define here your list of image file names 
    images = fddm.read_images(file_names)

    pixel_size = 0.3    # um
    frame_rate = 50     # frames per second
    
    # compute image structure function and set experimental parameters
    dqt = fddm.ddm(img_seq, range(1, len(img_seq)))
    dqt.pixel_size = pixel_size
    dqt.set_frame_rate(frame_rate)

    # compute the azimuthal average
    aa = fddm.azimuthal_average(dqt, bins=dqt.shape[-1], range=(0.0, dqt.ky[-1]))


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
