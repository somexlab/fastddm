"""Testing all imports and check if they work."""


def test_init_import():
    from fastddm import tiff2numpy, images2numpy, read_images  # noqa: F401
    from fastddm import (  # noqa: F401
        ImageStructureFunction,
        AzimuthalAverage,
        ddm,
        azimuthal_average,
        melt,
        mergesort,
    )


def test_weights_import():
    from fastddm.weights import sector_average_weight, sphere_form_factor  # noqa: F401


def test_window_import():
    from fastddm.window import blackman, blackman_harris  # noqa: F401


def test_mask_import():
    from fastddm.mask import central_cross_mask  # noqa: F401


def test_lags_import():
    from fastddm.lags import logspace_int, fibonacci  # noqa: F401


def test_fit_import():
    from fastddm.fit import (  # noqa: F401
        simple_exp_model,
        simple_structure_function,
        fit,
    )


def test_utils_import():
    from fastddm._utils import (  # noqa: F401
        read_metadata,
        read_images,
        images2numpy,
        tiff2numpy,
    )


def test_io_import():
    from fastddm._io import _store_data, load, _save_as_tiff  # noqa: F401
