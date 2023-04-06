"""Test the different implementations of ddm.

Author: Fabian Krautgasser | fkrautgasser@posteo.org
"""

from pathlib import Path

import pytest
import fastddm as fddm
import numpy as np

impath = Path("tests/test-imgs/confocal/")
imgs = fddm.read_images([p for p in sorted(impath.glob("*.tif"))][:20])
lags = np.arange(1, 10)

CORES = list(fddm._ddm._backend.keys())


@pytest.fixture
def ddm_baseline():
    return fddm.ddm(imgs, lags, core="cpp", mode="fft")


def test_ddm_cpp_fft(regtest, ddm_baseline):

    # recreate full spectrum to check against regression file
    full_shape = ddm_baseline.full_shape()
    full = np.zeros((full_shape[0]+2, *full_shape[1:]))
    for i in range(full_shape[0]):
        full[i] = ddm_baseline.full_slice(i)
    full[-2] = ddm_baseline.full_power_spec()
    full[-1] = ddm_baseline.full_var()

    with np.printoptions(threshold=np.inf):  # type: ignore
        print(full, file=regtest)


def test_ddm_cpp_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cpp", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()


def test_ddm_py_fft(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="fft")

    assert np.isclose(result._data, ddm_baseline._data).all()


def test_ddm_py_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_fft(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="fft")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()
