# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

from typing import Union, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np


@dataclass
class ContinResult:
    alpha: float
    alpha_over_s1: float
    obj_fctn: float
    var: float
    std: float
    deg_freedom: float
    prob1_to_reject: float
    prob2_to_reject: float
    lambda_m: np.ndarray
    c_m: np.ndarray
    err_c_m: np.ndarray
    tau: np.ndarray
    fit: np.ndarray
    residuals: np.ndarray

    def recompute_input(self) -> np.ndarray:
        """Recompute the input data

        Returns
        -------
        np.ndarray
            The input data (approx.)
        """
        return self.fit + self.residuals
