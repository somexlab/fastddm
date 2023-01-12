"""Main function to interface with backends."""

import numpy as np
from typing import Iterable, Dict, Callable, Union, Optional, Tuple
from functools import partial
from dataclasses import dataclass

IS_CPP_ENABLED = ${IS_CPP_ENABLED}      # configured by CMake
IS_CUDA_ENABLED = ${IS_CUDA_ENABLED}    # configured by CMake

from ._dfm_python import _py_image_structure_function

## setup of backend dictionary, initially only with py backend
_backend: Dict[str, Dict[str, Callable]] = {
    "py": {
        "direct": partial(_py_image_structure_function, mode="direct"),
        "fft": partial(_py_image_structure_function, mode="fft"),
    }
}

from ._fftopt import next_fast_len

if IS_CPP_ENABLED:
    from ._dfm_cpp import dfm_direct_cpp, dfm_fft_cpp
    # enable cpp support in backends
    _backend["cpp"] = {
        "direct": dfm_direct_cpp,
        "fft": dfm_fft_cpp
    }

if IS_CUDA_ENABLED:
    from ._dfm_cuda import dfm_direct_gpu, dfm_fft_gpu
    # enable cuda support in backends
    _backend["cuda"] = {
        "direct": dfm_direct_gpu,
        "fft": dfm_fft_gpu
    }


@dataclass
class ImageStructureFunction:
    """Image structure function container class.

    Parameters
    ----------
    data : np.ndarray
        The 2D image structure function.
    shape : Tuple[int, int, int]
        The shape of the image structure function data.
    kx : np.ndarray
        The array of wavevector values over x.
    ky : np.ndarray
        The array of wavevector values over y.
    tau : np.ndarray
        The array of time delays.
    pixel_size : float
        The image effective pixel size.
    delta_t : float
        The time delay between two consecutive images.
    """

    data : np.ndarray
    kx : np.ndarray
    ky : np.ndarray
    tau : np.ndarray
    pixel_size : float = 1.0
    delta_t : float = 1.0
    shape : Tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self):
        """Perform post init operations
        """
        self.shape = self.data.shape

    def set_pixel_size(self, pixel_size : float):
        """Set the image effective pixel size.

        This will propagate also on the values of kx and ky.

        Parameters
        ----------
        pixel_size : float
            The effective pixel size.
        """
        self.pixel_size = pixel_size
        dimy, dimx = self.data.shape[1:]
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dimx, d=pixel_size))
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dimy, d=pixel_size))

    def set_delta_t(self, delta_t : float):
        """Set the time delay between two consecutive frames.

        This will propagate also on the values of tau.

        Parameters
        ----------
        delta_t : float
            The time delay.
        """
        self.tau *= delta_t / self.delta_t
        self.delta_t = delta_t

    def set_frame_rate(self, frame_rate : float):
        """Set the acquisition frame rate.

        This will propagate also on the values of tau.

        Parameters
        ----------
        frame_rate : float
            The acquisition frame rate.
        """
        self.set_delta_t(1 / frame_rate)

    def shape(self) -> Tuple[int, int, int]:
        """Get the shape of the image structure function data

        Returns
        -------
        Tuple[int, int, int]
            Shape of image structure function
        """
        return self.data.shape


def ddm(
    img_seq: np.ndarray,
    lags: Iterable[int],
    *,
    core: str = "py",
    mode: str = "fft",
    **kwargs,
) -> np.ndarray:
    """Perform DDM analysis on given image sequence.
    Returns the full image structure function.

    Parameters
    ----------
    img_seq : np.ndarray
        Image sequence of shape (t, y, x) where t is time.
    lags : Iterable[int]
        The delays to be inspected by the analysis.
    core : str, optional
        The backend core, choose between "py", "cpp", and "cuda".
        Default is "py".
    mode : str, optional
        The mode of calculating the autocorrelation, choose between "direct"
        and "fft". Default is "fft".

    Returns
    -------
    np.ndarray
        The normalized image structure function.

    Raises
    ------
    RuntimeError
        If a value for `core` other than "py", "cpp", and "cuda" are given.
    RuntimeError
        If a value for `mode` other than "direct" and "fft" are given.
    """

    # renaming for convenience
    backend = _backend
    cores = list(backend.keys())
    modes = ["direct", "fft"]

    # sanity checks
    if core not in cores:
        raise RuntimeError(
            f"Unknown core '{core}' selected. Only possible options are {cores}."
        )

    if mode not in modes:
        raise RuntimeError(
            f"Unknown mode '{mode}' selected. Only possible options are {modes}."
        )

    # throw out duplicates and sort lags in ascending order
    lags = np.array(sorted(set(lags)))

    # read actual image dimensions
    dim_t, dim_y, dim_x = img_seq.shape

    # dimensions after zero padding for efficiency and for normalization
    dim_x_padded = next_fast_len(dim_x, core)
    dim_y_padded = next_fast_len(dim_y, core)

    # select backend
    ddm_func = backend[core][mode]

    # setup arguments
    if core != "py":
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

        if mode == "fft":
            dim_t_padded = next_fast_len(2 * dim_t, core)
            args.append(dim_t_padded)

    else:
        args = [img_seq, lags, dim_x_padded, dim_y_padded]

    kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_x_padded))
    ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_y_padded))
    return ImageStructureFunction(ddm_func(*args, **kwargs), kx, ky, lags.astype(np.float64))


def azimuthal_average(
    data : Union[ImageStructureFunction, np.ndarray],
    kx : Optional[np.ndarray] = None,
    ky : Optional[np.ndarray] = None,
    bins : Optional[Union[int,Iterable[float]]] = 10,
    range : Optional[Tuple[float, float]] = None,
    mask : Optional[np.ndarray] = None,
    weights : Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the azimuthal average of the image structure function.
    Returns the azimuthal average and the bin centers.

    Parameters
    ----------
    data : Union[ImageStructureFunction, np.ndarray]
        The image structure function.
    kx : np.ndarray, optional
        The array of spatial frequencies along axis x. If kx is None
        and data is not an ImageStructureFunction object the frequencies
        evaluated with `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Nx))`
        are used (`Nx = data.shape[2]`). Default is None.
    ky : np.ndarray, optional
        The array of spatial frequencies along axis y. If ky is None
        and data is not an ImageStructureFunction object the frequencies
        evaluated with `2.0 * np.pi * np.fft.fftshift(np.fft.fftfreq(Ny))`
        are used (`Ny = data.shape[1]`). Default is None.
    bins : Union[int, Iterable[float]], optional
        If `bins` is an int, it defines the number of equal-width bins in the
        given range (10, by default). If `bins` is a sequence, it defines a
        monotonically increasing array of bin edges, including the rightmost
        edge, allowing for non-uniform bin widths.
    range : (float, float), optional
        The lower and upper range of the bins. If not provided, range is simply
        `(k.min(), k.max())`, where `k` is the vector modulus computed from
        `kx` and `ky`. Values outside the range are ignored. The first element
        of the range must be less than or equal to the second.
    mask : np.ndarray, optional
        If a boolean `mask` is given, it is used to exclude grid points from
        the azimuthal average (where False is set). The array must have the
        same y,x shape of `data`.
    weights : np.ndarray, optional
        An array of weights, of the same y,x shape as `data`. Each
        value in `data` only contributes its associated weight towards
        the bin count (instead of 1).

    Returns
    -------
    az_avg : np.ndarray
        The azimuthal average.
    k : np.ndarray
        The spatial frequency associated to the bin.
    bin_edges : np.ndarray
        The bin edges.
    """

    # read actual image structure function shape
    dim_t, dim_y, dim_x = data.shape

    if isinstance(data, ImageStructureFunction):
        kx = data.kx
        ky = data.ky
    else:
        # check kx and ky
        if kx is None:
            kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_x))
        if ky is None:
            ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_y))

    # compute the k modulus
    X, Y = np.meshgrid(kx, ky)
    k_modulus = np.sqrt(X ** 2 + Y ** 2)

    # check range
    if range is None:
        k_min = np.min(k_modulus)
        k_max = np.max(k_modulus)
    else:
        k_min, k_max = range

    # check mask
    if mask is None:
        mask = np.full((dim_y, dim_x), True)

    # check weights
    if weights is None:
        weights = np.ones((dim_y, dim_x), dtype=np.float64)

    # compute bin edges and initialize k
    if isinstance(bins, int):
        bin_edges = np.linspace(k_min, k_max, bins)
        k = np.zeros(bins, dtype=np.float64)
    else:   # bins is an iterable
        bin_edges = [k_min]
        for bin_width in bins:
            bin_edges.append(bin_edges[-1] + bin_width)
        k = np.zeros(len(bins), dtype=np.float64)
        bins = len(bins) + 1

    # initialize azimuthal average
    az_avg = np.zeros((bins, dim_t), dtype=np.float64)

    # loop over bins
    for i, curr_bin_edge in enumerate(bin_edges):
        if i > 0:
            e_inf = bin_edges[i-1]
            e_sup = curr_bin_edge
            curr_px = (k_modulus > e_inf) & (k_modulus <= e_sup) & mask
        else:
            curr_px = (k_modulus == curr_bin_edge) & mask

        if np.all(np.logical_not(curr_px)):
            az_avg[i] = np.full(dim_t, np.nan)
            if i > 0:
                e_inf = bin_edges[i-1]
                e_sup = curr_bin_edge
                k[i] = (e_inf + e_sup) / 2.
            else:
                k[0] = curr_bin_edge
        else:
            num = (k_modulus[curr_px] * weights[curr_px]).mean()
            den = weights[curr_px].mean()
            k[i] = num / den
            if isinstance(data, ImageStructureFunction):
                w_avg = (data.data[:, curr_px] * weights[curr_px]).mean(axis=-1)
            else:
                w_avg = (data[:, curr_px] * weights[curr_px]).mean(axis=-1)
            az_avg[i] = w_avg / den

    return az_avg, k, bin_edges
