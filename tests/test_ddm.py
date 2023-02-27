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

CORES = list(fddm._backend.keys())


@pytest.fixture
def ddm_baseline():
    return fddm.ddm(imgs, lags, core="cpp", mode="fft")


def test_ddm_cpp_fft(regtest, ddm_baseline):

    with np.printoptions(threshold=np.inf):  # type: ignore
        print(ddm_baseline._data, file=regtest)


def test_ddm_cpp_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cpp", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()


def test_ddm_py_fft(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="fft")

    assert np.isclose(result._data, ddm_baseline.data).all()


def test_ddm_py_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="diff")

    assert np.isclose(result._data, ddm_baseline.data).all()


@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_fft(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="fft")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()
