"""Testing the _utils.py file."""
import pytest

from fastddm._utils import read_metadata

def test_read_metadata():
    path = "tests/test-imgs/nikon/test.nd2"

    # check .nd2 reading - we only assume that a dict is returned
    metadata = read_metadata(path)
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
    
