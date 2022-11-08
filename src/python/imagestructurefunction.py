from dataclasses import dataclass, field
from typing import List, Optional, Union

import numpy as np


@dataclass
class ImageStructureFunction:
    """Dataclass to store the Image Structure function and related quantities.

    Parameters
    ----------
    data : np.ndarray
        The full image structure function of shape (N, y, x).
    lags : np.ndarray
        The corresponding lags of length N in frame-units.
    taus : np.ndarray
        The physical lags, i.e. the lags divided by the fps, of length N.
    dt : Union[float, int]
        The conversion from frames to physical time, i.e. inverse of the fps.
    kx : np.ndarray
        FFT frequencies in x direction in units of inverse pixel.
    ky : np.ndarray
        FFT frequencies in y direction in units of inverse pixel.
    qx : np.ndarray
        Wave vectors in x direction in image appropriate units of inverse length.
    qy : np.ndarray
        Wave vectors in y direction in image appropriate units of inverse length.
    pixel_size : Union[float, int]
        The effective pixel size in length units per pixel.
    tau_units: Optional[str]
        Units of physical time. (to be used e.g. in plots)
    q_units : Optional[str]
        Units of physical wave vectors. (to be used e.g. in plots)
    pixel_units : Optional[str]
        Units of the `pixel_size` property. (to be used e.g. in plots)
    comments: Optional[str]
        Notes regarding the data or the experiment.
    log: List[str]
        A list of log messages that describe the computational steps for this data set.
    """

    data: np.ndarray
    lags: np.ndarray
    taus: np.ndarray
    dt: Union[float, int]
    kx: np.ndarray
    ky: np.ndarray
    qx: np.ndarray
    qy: np.ndarray
    pixel_size: Union[float, int]
    tau_units: Optional[str] = None
    q_units: Optional[str] = None
    pixel_units: Optional[str] = None
    comments: Optional[str] = None
    log: List[str] = field(default_factory=list)
