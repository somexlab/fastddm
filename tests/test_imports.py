"""Testing all imports and check if they work."""


def test_init_import():
    """Test imports from fastddm."""
    from fastddm import (  # noqa: F401,
        azimuthal_average,
        ddm,
        images2numpy,
        lags,
        load,
        mask,
        read_images,
        tiff2numpy,
        weights,
        window,
    )


def test_weights_import():
    """Test imports from fastddm.weights."""
    from fastddm.weights import sector_average_weight, sphere_form_factor  # noqa: F401


def test_window_import():
    """Test imports from fastddm.window."""
    from fastddm.window import blackman, blackman_harris  # noqa: F401


def test_mask_import():
    """Test imports from fastddm.mask."""
    from fastddm.mask import central_cross_mask  # noqa: F401


def test_lags_import():
    """Test imports from fastddm.lags."""
    from fastddm.lags import fibonacci, logspace_int  # noqa: F401


def test_fit_import():
    """Test imports from fastddm.fit."""
    from fastddm.fit import (  # noqa: F401
        fit,
        simple_exp_model,
        simple_structure_function,
    )


def test_utils_import():
    """Test imports from fastddm.utils."""
    from fastddm.utils import (  # noqa: F401
        images2numpy,
        read_images,
        read_metadata,
        tiff2numpy,
    )


def test_io_common_import():
    """Test imports from fastddm._io_common."""
    from fastddm._io_common import (  # noqa: F401
        Parser,
        Reader,
        Writer,
        _save_as_tiff,
        calculate_format_size,
        npdtype2format,
    )
