# `fastddm` :rocket:

## About
Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.
## Dependencies
All necessary dependencies will be installed automatically. However, if you also want to use the `dfmtoolbox.fit` module, you have to install `lmfit` (e.g. with `pip install lmfit`).
## Installation
To install `fastddm` from source, first clone this repository locally (`git clone https://github.com/somexlab/dfmtoolbox.git`). It is recommended to use it in a virtual environment of your choice, e.g. `conda` (to create a new virtual environment with conda, do `conda create -n fastddm python=3.10` to create a virtual environment `fastddm`, using python 3.10.* as the interpreter.)

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
>>> import dfmtoolbox as ddm
>>> ddm.IS_CPP_ENABLED
True
>>> ddm.IS_CUDA_ENABLED
True
```
## Usage examples

## Citation
