"""Main function to interface with backends."""

from typing import Iterable, Dict, Callable, Union, Optional, Tuple, Sequence
from functools import partial
from dataclasses import dataclass
import pickle
import os.path
import numpy as np
from scipy.interpolate import interp1d

IS_CPP_ENABLED = ${IS_CPP_ENABLED}      # configured by CMake
IS_CUDA_ENABLED = ${IS_CUDA_ENABLED}    # configured by CMake

from ._io import load, _store_data, _save_as_tiff
from ._ddm_python import _py_image_structure_function

## setup of backend dictionary, initially only with py backend
_backend: Dict[str, Dict[str, Callable]] = {
    "py": {
        "diff": partial(_py_image_structure_function, mode="diff"),
        "fft": partial(_py_image_structure_function, mode="fft"),
    }
}

from ._fftopt import next_fast_len

if IS_CPP_ENABLED:
    from ._ddm_cpp import ddm_diff_cpp, ddm_fft_cpp
    # enable cpp support in backends
    _backend["cpp"] = {
        "diff": ddm_diff_cpp,
        "fft": ddm_fft_cpp
    }

if IS_CUDA_ENABLED:
    from ._ddm_cuda import ddm_diff_gpu, ddm_fft_gpu
    # enable cuda support in backends
    _backend["cuda"] = {
        "diff": ddm_diff_gpu,
        "fft": ddm_fft_gpu
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

    def set_pixel_size(self, pixel_size : float) -> None:
        """Set the image effective pixel size.

        This will propagate also on the values of kx and ky.

        Parameters
        ----------
        pixel_size : float
            The effective pixel size.
        """
        self.kx *= self.pixel_size / pixel_size
        self.ky *= self.pixel_size / pixel_size
        self.pixel_size = pixel_size

    def set_delta_t(self, delta_t : float) -> None:
        """Set the time delay between two consecutive frames.

        This will propagate also on the values of tau.

        Parameters
        ----------
        delta_t : float
            The time delay.
        """
        self.tau *= delta_t / self.delta_t
        self.delta_t = delta_t

    def set_frame_rate(self, frame_rate : float) -> None:
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

    def save(
        self,
        *,
        fname : str = "analysis_blob",
        protocol : int = pickle.HIGHEST_PROTOCOL
        ) -> None:
        """Save ImageStructureFunction to binary file.
        The binary file is in fact a python pickle file.

        Parameters
        ----------
        fname : str, optional
            The full file name, by default "analysis_blob".
        protocol : int, optional
            pickle binary serialization protocol, by default
            pickle.HIGHEST_PROTOCOL.
        """
        # check name
        dir, name = os.path.split(fname)
        name = name if name.endswith(".sf.ddm") else f"{name}.sf.ddm"

        # save to file
        _store_data(self, fname=os.path.join(dir, name), protocol=protocol)

    def save_as_tiff(
        self,
        seq : Sequence[int],
        fnames : Sequence[str]
        ) -> None:
        """Save ImageStructureFunction frames as images.

        Parameters
        ----------
        seq : Optional[Sequence[int]]
            List of indices to export.
        fnames : Optional[Sequence[str]], optional
            List of file names.

        Raises
        ------
        RuntimeError
            If number of elements in fnames and seq are different.
        """
        if len(fnames) != len(seq):
            raise RuntimeError('Number of elements in fnames differs from one in seq.')

        _save_as_tiff(data=self.data[seq], labels=fnames)


@dataclass
class AzimuthalAverage:
    """Azimuthal average container class.

    Parameters
    ----------
    data : np.ndarray
        The azimuthal average data.
    k : np.ndarray
        The array of reference wavevector values in the bins.
    tau : np.ndarray
        The array of time delay values.
    bin_edges : np.ndarray
        The array of bin edges.
    """

    data : np.ndarray
    k : np.ndarray
    tau : np.ndarray
    bin_edges : np.ndarray

    def save(
        self,
        *,
        fname : str = "analysis_blob",
        protocol : int = pickle.HIGHEST_PROTOCOL
        ) -> None:
        """Save AzimuthalAverage to binary file.
        The binary file is in fact a python pickle file.

        Parameters
        ----------
        fname : str, optional
            The full file name, by default "analysis_blob".
        protocol : int, optional
            pickle binary serialization protocol, by default
            pickle.HIGHEST_PROTOCOL.
        """
        # check name
        dir, name = os.path.split(fname)
        name = name if name.endswith(".aa.ddm") else f"{name}.aa.ddm"

        # save to file
        _store_data(self, fname=os.path.join(dir, name), protocol=protocol)

    def resample(
        self,
        tau : np.ndarray
        ) -> 'AzimuthalAverage':
        """Resample with new tau values and return a new AzimuthalAverage.

        Parameters
        ----------
        tau : np.ndarray
            New values of tau.

        Returns
        -------
        AzimuthalAverage
            The resampled azimuthal average.
        """
        # initialize data
        data = np.zeros((len(self.k),len(tau)))

        _tau = np.log(tau)

        # loop through k values
        for i in range(len(self.k)):
            # check for nan
            if np.isnan(self.data[i, 0]):
                data[i] = np.full(len(tau), np.nan)
            else:
                # interpolate points in loglog scale
                f = interp1d(
                    x=np.log(self.tau),
                    y=np.log(self.data[i]),
                    kind='quadratic',
                    fill_value='extrapolate'
                    )
                data[i] = np.exp(f(_tau))
        
        return AzimuthalAverage(data, self.k, tau, self.bin_edges)


def ddm(
    img_seq: np.ndarray,
    lags: Iterable[int],
    *,
    core: str = "py",
    mode: str = "fft",
    **kwargs,
    ) -> ImageStructureFunction:
    """Perform Differential Dynamic Microscopy analysis on given image sequence.
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
        The mode of calculating the autocorrelation, choose between "diff"
        and "fft". Default is "fft".

    Returns
    -------
    ImageStructureFunction
        The image structure function.

    Raises
    ------
    RuntimeError
        If a value for `core` other than "py", "cpp", and "cuda" are given.
    RuntimeError
        If a value for `mode` other than "diff" and "fft" are given.
    """

    # renaming for convenience
    backend = _backend
    cores = list(backend.keys())
    modes = ["diff", "fft"]

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
    tau : Optional[np.ndarray] = None,
    kx : Optional[np.ndarray] = None,
    ky : Optional[np.ndarray] = None,
    bins : Optional[Union[int,Iterable[float]]] = 10,
    range : Optional[Tuple[float, float]] = None,
    mask : Optional[np.ndarray] = None,
    weights : Optional[np.ndarray] = None
    ) -> AzimuthalAverage:
    """Compute the azimuthal average of the image structure function.

    Parameters
    ----------
    data : Union[ImageStructureFunction, np.ndarray]
        The image structure function.
    tau : np.ndarray, optional
        The array of time delay values. Required if data is not an
        ImageStructureFunction object. Default is None.
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
    AzimuthalAverage
        The azimuthal average.

    Raises
    ------
    ValueError
        If tau, kx, and ky are not compatible with shape of data.
    """

    # check input arguments
    if isinstance(data,ImageStructureFunction):
        tau = data.tau
        kx = data.kx
        ky = data.ky
    else:
        if tau is None:
            raise ValueError("`tau` must be given for non-`ImageStructureFunction` data input.")
        elif len(tau) != len(data):
            raise ValueError("Length of `tau` not compatible with shape of `data`.")

    # read actual image structure function shape
    dim_t, dim_y, dim_x = data.shape

    # check kx and ky
    if kx is None:
        kx = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_x))
    elif len(kx) != data.shape[2]:
        raise ValueError("Length of `kx` not compatible with shape of `data`.")
    if ky is None:
        ky = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(dim_y))
    elif len(ky) != data.shape[1]:
        raise ValueError("Length of `ky` not compatible with shape of `data`.")

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

    return AzimuthalAverage(az_avg, k, tau.astype(np.float64), bin_edges)


def melt(
    az_avg1 : AzimuthalAverage,
    az_avg2 : AzimuthalAverage
    ) -> AzimuthalAverage:
    """Melt two azimuthal averages into one object.

    Parameters
    ----------
    az_avg1 : AzimuthalAverage
        One AzimuthalAverage object.
    az_avg2 : AzimuthalAverage
        Another AzimuthalAverage object.

    Returns
    -------
    AzimuthalAverage
        The two AzimuthalAverage objects, merged into a new one.
    """

    # assign fast and slow acquisition
    if az_avg1.tau[0] < az_avg2.tau[0]:
        fast, slow = az_avg1, az_avg2
    else:
        fast, slow = az_avg2, az_avg1
    
    # initialize tau and data
    Nt = min(10, np.sum(slow.tau < fast.tau[-1]))
    # keep data of `fast` up to (Nt/2)-th tau of `slow`
    idx = np.argmin(np.abs(fast.tau - slow.tau[Nt // 2])) + 1
    tau = np.append(fast.tau[:idx], slow.tau[Nt // 2 + 1:])
    data = np.zeros((len(fast.k), len(tau)))

    t = np.log(slow.tau[:Nt])

    # loop through k values
    for i in range(len(fast.k)):
        # check for nan
        if np.any(np.isnan([fast.data[i, 0], slow.data[i, 0]])):
            data[i] = np.full(len(tau), np.nan)
        else:
            # find multiplicative factor via least squares minimization
            # interpolate in loglog scale (smoother curve)
            f = interp1d(x=np.log(fast.tau), y=np.log(fast.data[i]), kind='cubic')
            sum_xiyi = np.sum(np.exp(f(t)) * slow.data[i, :Nt])
            sum_xi2 = np.sum(np.exp(f(t)) ** 2)
            alpha = sum_xiyi / sum_xi2
            
            # scale fast on slow
            data[i] = np.append(fast.data[i, :idx] * alpha, slow.data[i, Nt // 2 + 1:])

    return AzimuthalAverage(data, fast.k, tau, fast.bin_edges)


def mergesort(
    az_avg1 : AzimuthalAverage,
    az_avg2 : AzimuthalAverage
    ) -> AzimuthalAverage:
    """Merge the values of two azimuthal averages.
    Values will then be sorted based on tau.

    Parameters
    ----------
    az_avg1 : AzimuthalAverage
        One AzimuthalAverage object.
    az_avg2 : AzimuthalAverage
        Another AzimuthalAverage object.

    Returns
    -------
    AzimuthalAverage
        The two AzimuthalAverage objects are fused into a new one.
    """
    tau = np.append(az_avg1.tau, az_avg2.tau)
    data = np.append(az_avg1.data, az_avg2.data, axis=1)
    sortidx = np.argsort(tau)
    return AzimuthalAverage(data[:,sortidx], az_avg1.k, tau[sortidx], az_avg1.bin_edges)