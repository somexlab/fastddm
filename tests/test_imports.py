"""Testing all imports and check if they work.

Author: Fabian Krautgasser | fkrautgasser@posteo.org
"""


def test_init_import():
    from fastddm import (  # noqa: F401,
        tiff2numpy,
        images2numpy,
        read_images,
        ddm,
        azimuthal_average,
        load,
        lags,
        mask,
        weights,
        window,
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
    from fastddm.utils import (  # noqa: F401
        read_metadata,
        read_images,
        images2numpy,
        tiff2numpy,
    )


def test_io_common_import():
    from fastddm._io_common import (  # noqa: F401
        calculate_format_size,
        npdtype2format,
        Writer,
        Reader,
        Parser,
        _save_as_tiff,
    )
