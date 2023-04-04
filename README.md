# `FastDDM` :rocket:
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**FastDDM** is a Python package for the analysis of microscopy image sequences using Differential Dynamic Microscopy on CPU and GPU.
The features implemented are targeted at the experimental soft matter research community dealing with inert and active/biological samples.

## Dependencies
All necessary dependencies will be installed automatically. However, if you also want to use the `fastddm.fit` module, you have to manually install `lmfit` (e.g. with `pip install lmfit`).

## Resources
Links to readthedocs pages...

## Installation
To install `fastddm` from source, first clone this repository locally (`git clone https://github.com/somexlab/fastddm.git`). It is recommended to use it in a virtual environment of your choice, e.g. `conda` (to create a new virtual environment with conda, do `conda create -n fastddm python=3.10` to create a virtual environment `fastddm`, using python 3.10.* as the interpreter.)

### (i) installing `fastddm` only with python backend
In your preferred environment, run from the local git clone
```console
pip install .
```

### (ii) installing `fastddm` with additional C++ backend
In your preferred environment, run from the local git clone
```console
export ENABLE_CPP=ON
pip install .
```
### (iii) installing `fastddm` with CUDA support
Note: this will also install the C++ backend.

In your preferred environment, run from the local git clone
```console
export ENABLE_CUDA=ON
pip install .
```
### Check your installation of C++ / CUDA is working
In a python shell run:
```python
>>> import fastddm as fddm
>>> fddm.IS_CPP_ENABLED
True
>>> fddm.IS_CUDA_ENABLED
True
```
## Usage examples
Check the examples folder for jupyter notebooks!

## Contributing to FastDDM
Contributions are welcome via [pull requests](https://github.com/somexlab/fastddm/pulls).
Please, report bugs and suggest features via the [issue tracker](https://github.com/somexlab/fastddm/issues).

## Citing FastDDM
Please, cite this publication in every work that uses FastDDM:

    E. Lattuada, F. Krautgasser, F. Giavazzi, and R. Cerbino.
    The hitchhiker's guide to Differential Dynamic Microscopy.
    The Journal of Chemical Physics XX: YYYY, ZZZ 2023.
    `10.1016/j.chem.phys. blablabla <https://...>`

## License
FastDDM is available under the [GNU GPL-3.0 license](LICENSE).

## Acknowledgements

* The [fftw-3.3.10](https://www.fftw.org/) and [pybind11 2.10.0](https://github.com/pybind/pybind11) libraries are included in the source tree.
* This project was founded by the Austrian Science Fund (FWF), Grant Number M 3250-N.
