"""Test the different implementations of ddm.

Author: Fabian Krautgasser | fkrautgasser@posteo.org
"""

from pathlib import Path

import pytest
import fastddm as fddm
import numpy as np

from fastddm import IS_SINGLE_PRECISION, DTYPE

impath = Path("tests/test-imgs/confocal/")
imgs = fddm.read_images([p for p in sorted(impath.glob("*.tif"))][:20])
lags = np.arange(1, 10)

CORES = list(fddm._ddm._backend.keys())


@pytest.fixture
def ddm_baseline():
    return fddm.ddm(imgs, lags, core="cpp", mode="fft")


@pytest.mark.skipif(
    IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option"
)
def test_ddm_cpp_fft(regtest, ddm_baseline):
    # recreate full spectrum to check against regression file
    full_shape = ddm_baseline.full_shape()
    full = np.zeros((full_shape[0] + 2, *full_shape[1:]))
    for i in range(full_shape[0]):
        full[i] = ddm_baseline.full_slice(i)
    full[-2] = ddm_baseline.full_power_spec()
    full[-1] = ddm_baseline.full_var()

    with np.printoptions(threshold=np.inf):  # type: ignore
        print(full, file=regtest)


@pytest.mark.skipif(
    IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option"
)
def test_ddm_cpp_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cpp", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif(
    IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option"
)
def test_ddm_py_fft(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="fft")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif(
    IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option"
)
def test_ddm_py_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif(
    IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option"
)
@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_fft(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="fft")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif(
    IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option"
)
@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_diff(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data).all()


@pytest.mark.skipif(
    not IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option OFF"
)
def test_ddm_cpp_fft_single(regtest, ddm_baseline):
    # recreate full spectrum to check against regression file
    full_shape = ddm_baseline.full_shape()
    full = np.zeros((full_shape[0] + 2, *full_shape[1:]))
    for i in range(full_shape[0]):
        full[i] = ddm_baseline.full_slice(i)
    full[-2] = ddm_baseline.full_power_spec()
    full[-1] = ddm_baseline.full_var()

    with np.printoptions(threshold=np.inf):  # type: ignore
        print(full, file=regtest)


@pytest.mark.skipif(
    not IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option OFF"
)
def test_ddm_cpp_diff_single(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cpp", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data, atol=0.0, rtol=1e-3).all()


@pytest.mark.skipif(
    not IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option OFF"
)
def test_ddm_py_fft_single(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="fft")

    assert np.isclose(result._data, ddm_baseline._data, atol=0.0, rtol=1e-3).all()


@pytest.mark.skipif(
    not IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option OFF"
)
def test_ddm_py_diff_single(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="py", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data, atol=0.0, rtol=1e-3).all()


@pytest.mark.skipif(
    not IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option OFF"
)
@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_fft_single(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="fft")

    assert np.isclose(result._data, ddm_baseline._data, atol=0.0, rtol=1e-3).all()


@pytest.mark.skipif(
    not IS_SINGLE_PRECISION, reason="installed with SINGLE_PRECISION option OFF"
)
@pytest.mark.skipif("cuda" not in CORES, reason="needs CUDA installed")
def test_ddm_cuda_diff_single(ddm_baseline):
    result = fddm.ddm(imgs, lags, core="cuda", mode="diff")

    assert np.isclose(result._data, ddm_baseline._data, atol=0.0, rtol=1e-3).all()


def test_ddm_lags_errors():
    # check error when negative lags are given
    with pytest.raises(RuntimeError):
        fddm.ddm(imgs, [-1], core="py", mode="diff")


def test_ddm_window_error():
    # check error when window and lags are incompatible
    with pytest.raises(RuntimeError):
        dim_t, dim_y, dim_x = imgs.shape
        # make window slightly larger
        window = np.ones((dim_y + 1, dim_x), dtype=DTYPE)
        fddm.ddm(imgs, lags, core="cpp", mode="fft", window=window)
