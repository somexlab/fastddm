# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada and Fabian Krautgasser
# Maintainers: Enrico Lattuada and Fabian Krautgasser

"""Image structure function data class."""

from typing import Sequence, Tuple
from dataclasses import dataclass
import pickle
import os
import numpy as np

from ._io import _store_data, _save_as_tiff


@dataclass
class ImageStructureFunction:
    """Image structure function container class.

    Parameters
    ----------
    _data : np.ndarray
        The packed data (2D image structure function, power spectrum,
        and variance).
    kx : np.ndarray
        The array of wavevector values over x.
    ky : np.ndarray
        The array of wavevector values over y.
    tau : np.ndarray
        The array of time delays.
    _pixel_size : float, optional
        The effective pixel size. Default is 1.
    _delta_t : float, optional
        The time delay between two consecutive frames. Default is 1.

    Attributes
    ----------
    data : np.ndarray
        The 2D image structure function.
    power_spec: np.ndarray
        The average 2D power spectrum of the input images.
    var : np.ndarray
        The 2D variance (over time) of the Fourier transformed images.
    kx : np.ndarray
        The array of wavevector values over x.
    ky : np.ndarray
        The array of wavevector values over y.
    tau : np.ndarray
        The array of time delays.
    pixel_size : float
        The effective pixel size.
    delta_t : float
        The time delay between to consecutive frames.
    shape : Tuple[int, int, int]
        The shape of the 2D image structure function.

    Methods
    -------
    set_frame_rate(frame_rate) : None
        Set the acquisition frame rate. This will propagate also on the values
        of tau.
    save(*, fname, protocol) : None
        Save ImageStructureFunction to binary file.
    save_as_tiff(seq, fnames) : None
        Save ImageStructureFunction frames as tiff images.
    """

    _data : np.ndarray
    kx : np.ndarray
    ky : np.ndarray
    tau : np.ndarray
    _pixel_size : float = 1.0
    _delta_t : float = 1.0

    @property
    def data(self) -> np.ndarray:
        """The 2D image structure function.

        Returns
        -------
        np.ndarray
            The 2D image structure function.
        """
        return self._data[:-2]

    @property
    def power_spec(self) -> np.ndarray:
        """The average 2D power spectrum of the input images.

        Returns
        -------
        np.ndarray
            The average 2D power spectrum of the input images.
        """
        return self._data[-2]

    @property
    def var(self) -> np.ndarray:
        """The variance (over time) of the Fourier transformed input images.

        Returns
        -------
        np.ndarray
            The variance (over time) of the Fourier transformed input images.
        """
        return self._data[-1]

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Shape of image structure function data.

        Returns
        -------
        Tuple[int, int, int]
            The shape of the data. 
        """
        return self.data.shape

    @property
    def pixel_size(self) -> float:
        """The effective pixel size.

        Returns
        -------
        float
            Pixel size.
        """
        return self._pixel_size

    @property
    def delta_t(self) -> float:
        """The time delay between to consecutive frames.

        Returns
        -------
        float
            Time delay.
        """
        return self._delta_t

    @pixel_size.setter
    def pixel_size(self, pixel_size : float) -> None:
        """Set the image effective pixel size.

        This will propagate also on the values of kx and ky.

        Parameters
        ----------
        pixel_size : float
            The effective pixel size.
        """
        self.kx *= self._pixel_size / pixel_size
        self.ky *= self._pixel_size / pixel_size
        self._pixel_size = pixel_size

    @delta_t.setter
    def delta_t(self, delta_t : float) -> None:
        """Set the time delay between two consecutive frames.

        This will propagate also on the values of tau.

        Parameters
        ----------
        delta_t : float
            The time delay.
        """
        self.tau *= delta_t / self._delta_t
        self._delta_t = delta_t

    def __len__(self):
        """The length of the image structure function data.
        It coincides with the number of lags.

        Returns
        -------
        int
            The length of data.
        """
        return len(self.data)

    def set_frame_rate(self, frame_rate : float) -> None:
        """Set the acquisition frame rate.

        This will propagate also on the values of tau.

        Parameters
        ----------
        frame_rate : float
            The acquisition frame rate.
        """
        self.delta_t = 1 / frame_rate

    def save(
        self,
        fname : str = "analysis_blob",
        *,
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
