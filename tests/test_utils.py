"""Testing the _utils.py file."""
from pathlib import Path
import pytest
import numpy as np

from fastddm._utils import read_metadata, read_images

ND2_TEST = "tests/test-imgs/nikon/test.nd2"
ND2_DIMS = (15000, 256, 256)
ND2_DTYPE = np.uint16

@pytest.fixture
def nd2_path():
    return ND2_TEST


@pytest.mark.skipif(not Path(ND2_TEST).exists(), reason="ND2 testfile not available.")
def test_read_metadata(nd2_path):

    # check .nd2 reading - we only assume that a dict is returned
    metadata = read_metadata(nd2_path)
    assert isinstance(metadata, dict)

    # check error when file is not found
    with pytest.raises(FileNotFoundError):
        read_metadata("nonexisting/path.tif")

    # check folder behaviour
    with pytest.raises(NotImplementedError):
        read_metadata("tests/test-imgs/confocal/")

    # check .tif behaviour
    with pytest.raises(RuntimeError):
        read_metadata("tests/test-imgs/confocal/0000.tif")


@pytest.mark.skipif(not Path(ND2_TEST).exists(), reason="ND2 testfile not available.")
def test_read_nd2(nd2_path):

    # read full nd2 file
    data = read_images(nd2_path)
    assert data.dtype == ND2_DTYPE
    assert data.shape == ND2_DIMS

    # read partial sequence
    start, stop = 50, 100
    data_part = read_images(nd2_path, seq=range(start, stop))
    assert (data[start: stop] == data_part).all()
