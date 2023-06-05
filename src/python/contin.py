# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

from typing import Union, Optional, Tuple, List, Dict
import os
import re
import numpy as np

from .contin_result import ContinResult

_DATA_FORMAT = ">15.8E"
_DATA_FORMAT_FORTRAN = "(1E15.8)"
_ALPHA_SECTION_START = "      ALPHA"
_CM_SECTION_START = "    ORDINATE    ERROR  ABSCISSA"
_CM_SECTION_END = "0LINEAR COEFFICIENTS"
_FIT_SECTION_START = "    ORDINATE  ABSCISSA"
_FIT_SECTION_END = "1"
_INPUT_DATA_SECTION_START1 = "            T            Y"
_INPUT_DATA_SECTION_START2 = "                 T            Y"
_INPUT_DATA_SECTION_END = "0PRECIS"

def _fortran2py_num_fmt(value: str) -> float:
    return float(value.replace('D', 'E'))


class Contin:
    """Wrapper class to CONTIN executable

    Attributes
    ----------
    k_cmm1 : float
        Wave vector (in cm^-1), by default 1
    tau_s : np.ndarray
        Delay times (in s), by default None
    isf : np.ndarray
        Intermediate scattering function data, by default None
    err : np.ndarray
        Intermediate scattering function uncertainty, by default None
    file_in : str
        Path to input file, by default "contin.in"
    file_out : str
        Path to output file, by default "contin.out"
    exec_path : str
        Path to CONTIN executable file
    mode : str
        Analysis mode, by default "time"
    viscosity_cP : float
        Solvent dynamic viscosity (in cP == mPa.s), by default 1
    temperature_K : float
        Absolute temperature (in K), by default 300
    refractive index : float
        Solvent refractive index, by default 1.333
    wavelength_nm : float
        Illumination wavelength (in nm), by default 550
    hollow_sphere_wall_thickness_cm : float
        Hollow sphere model wall thickness (in cm), by default -1
    norder : int
        Regularizor order, by default 2
    print_residuals : int
        Frequency of residuals print in output file mode, by default 3
    print_fits : int
        Frequency of fits print in output file mode, by default 3
    num_quad_grid_points : int
        Number of quadrature grid points, by default 40
    quad_grid_range : Tuple[float, float]
        Range of quadrature grid, by default None
    modes : List[str]
        Supported analysis modes
    
    Methods
    -------
    set_data(k, tau, isf, err) : None
        Set data for analysis. k must be in cm^-1, tau in s
    gen_grid_range_time() : Tuple[float, float]
        Estimate quadrature grid range for distribution of relaxation times
    gen_grid_range_diffusion() : Tuple[float, float]
        Estimate quadrature grid range for distribution of diffusion
        coefficients
    gen_grid_range_radius() : Tuple[float, float]
        Estimate quadrature grid range for distribution of spheres
        satisfying the Stokes-Einstein relation
    write_input() : None
        Write Contin input file for analysis
    run_contin() : None
        Run CONTIN executable
    analyze() : None
        Analyze data with CONTIN
    """
    _k_cmm1: float = 1.0
    _tau_s: np.ndarray = None
    _isf: np.ndarray = None
    _err: np.ndarray = None
    _file_in: str = "contin.in"
    _file_out: str = "contin.out"
    _exec_path: str
    _mode: str = "time"
    _visc_cP: float = 1.0
    _temp_K: float = 300.0
    _refr_index: float = 1.333
    _lambda_nm: float = 550.0
    _hollow_sph_wall_cm: float = -1.0
    _norder: int = 2
    _print_res: int = 3
    _print_fit: int = 3
    _num_quad_grid_pts: int = 40
    _quad_grid_range: Tuple[float, float] = None
    _modes: List[str] = ["time", "diffusion", "radius"]

    def __init__(self):
        self.__post_init__()

    def __post_init__(self):
        """Sets the path to the CONTIN executable
        """
        this_file_path = os.path.abspath(os.path.dirname(__file__))
        self._exec_path = os.path.join(this_file_path, "contin")

    @property
    def k_cmm1(self) -> float:
        """Wave vector (in cm^-1)

        Returns
        -------
        float
            The wave vector
        """
        return self._k_cmm1
    
    @k_cmm1.setter
    def k_cmm1(self, k: float) -> None:
        """Set wave vector magnitude (in cm^-1)

        Parameters
        ----------
        k : float
            The wave vector magnitude. Must be positive.

        Raises
        ------
        ValueError
            If the wave vector magnitude is not positive
        """
        if k > 0:
            self._k_cmm1 = k
        else:
            raise ValueError("Wave vector magnitude must be positive")

    @property
    def tau_s(self) -> np.ndarray:
        """Delay times (in s)

        Returns
        -------
        np.ndarray
            The delay times
        """
        return self._tau_s

    @property
    def isf(self) -> np.ndarray:
        """Intermediate scattering function data

        Returns
        -------
        np.ndarray
            The intermediate scattering function data
        """
        return self._isf
    
    @property
    def err(self) -> np.ndarray:
        """Intermediate scattering function uncertainty

        Returns
        -------
        np.ndarray
            The intermediate scattering function uncertainty
        """
        return self._err

    @property
    def file_in(self) -> str:
        """Path to input file

        Returns
        -------
        str
            The path to the input file
        """
        return self._file_in
    
    @file_in.setter
    def file_in(self, file_in: str) -> None:
        """Set path to input file

        Parameters
        ----------
        file_in : str
            The path to the input file
        """
        self._file_in = file_in

    @property
    def file_out(self) -> str:
        """Path to output file

        Returns
        -------
        str
            The path to the output file
        """
        return self._file_out
    
    @file_out.setter
    def file_out(self, file_out: str) -> None:
        """Set path to output file

        Parameters
        ----------
        file_out : str
            The path to the output file
        """
        self._file_out = file_out

    @property
    def exec_path(self) -> str:
        """Path to CONTIN executable file

        Returns
        -------
        str
            The path to the CONTIN executable file
        """
        return self._exec_path
    
    @property
    def mode(self) -> str:
        """CONTIN analysis mode

        Returns
        -------
        str
            The CONTIN analysis mode
        """
        return self._mode
    
    @mode.setter
    def mode(self, mode: str) -> None:
        """Set CONTIN analysis mode

        Parameters
        ----------
        mode : str
            The CONTIN analysis mode.
            Possible modes are 'time', 'diffusion', 'radius'.

        Raises
        ------
        RuntimeError
            If mode is not supported.
        """
        if mode not in self._modes:
            msg = f"Unknown mode {mode}. Supported modes are {self._modes}"
            raise RuntimeError(msg)
        else:
            self._mode = mode

    @property
    def viscosity_cP(self) -> float:
        """Solvent dynamic viscosity (in cP == mPa.s)

        Returns
        -------
        float
            The solvent dynamic viscosity
        """
        return self._visc_cP
    
    @viscosity_cP.setter
    def viscosity_cP(self, eta: float) -> None:
        """Set solvent dynamic viscosity (in cP == mPa.s)

        Parameters
        ----------
        eta : float
            The solvent dynamic viscosity

        Raises
        ------
        ValueError
            If eta is not positive
        """
        if eta > 0:
            self._visc_cP = eta
        else:
            raise ValueError("Dynamic viscosity must be positive")
        
    @property
    def temperature_K(self) -> float:
        """Absolute temperature (in K)

        Returns
        -------
        float
            The absolute temperature
        """
        return self._temp_K
    
    @temperature_K.setter
    def temperature_K(self, T: float) -> None:
        """Set absolute temperature (in K)

        Parameters
        ----------
        T : float
            The absolute temperature

        Raises
        ------
        ValueError
            If T is not positive
        """
        if T > 0:
            self._temp_K = T
        else:
            raise ValueError("Absolute temperature must be positive")
        
    @property
    def refractive_index(self) -> float:
        """Solvent refractive index

        Returns
        -------
        float
            The solvent refractive index
        """
        return self._refr_index
    
    @refractive_index.setter
    def refractive_index(self, n: float) -> None:
        """Set solvent refractive index

        Parameters
        ----------
        n : float
            The solvent refractive index

        Raises
        ------
        ValueError
            If n is not positive
        """
        if n > 0:
            self._refr_index = n
        else:
            raise ValueError("Solvent refractive index must be positive")
        
    @property
    def wavelength_nm(self) -> float:
        """Illumination wavelength (in nm)

        Returns
        -------
        float
            The illumination wavelength
        """
        return self._lambda_nm
    
    @wavelength_nm.setter
    def wavelength_nm(self, wavelength: float) -> None:
        """Set illumination wavelength (in nm)

        Parameters
        ----------
        wavelength : float
            The illumination wavelength

        Raises
        ------
        ValueError
            If wavelength is not positive
        """
        if wavelength > 0:
            self._lambda_nm = wavelength
        else:
            raise ValueError("Illumination wavelength must be positive")
        
    @property
    def hollow_sphere_wall_thickness_cm(self) -> float:
        """Hollow sphere model wall thickness (in cm)

        Returns
        -------
        float
            The wall thickness
        """
        return self._hollow_sph_wall_cm
    
    @hollow_sphere_wall_thickness_cm.setter
    def hollow_sphere_wall_thickness_cm(self, wall_thickness: float) -> None:
        """Set hollow sphere model wall thickness (in cm)

        Parameters
        ----------
        wall_thickness : float
            The wall thickness. If a negative value is input,
            the solid sphere model is used.
        """
        self._hollow_sph_wall_cm = wall_thickness

    @property
    def norder(self) -> int:
        """Regularizor model

        Returns
        -------
        int
            The regularizor order
        """
        return self._norder
    
    @norder.setter
    def norder(self, norder: int) -> None:
        """Set regularizor order

        Parameters
        ----------
        norder : int
            The regularizor order. Accepted values are in the interval [0, 5].
            A value of 2 corresponds to the sum of squared second order
            derivative terms and is usually considered a smooth solution.

        Raises
        ------
        ValueError
            If norder is out of the accepted range
        """
        if norder in range(6):
            self._norder = norder
        else:
            msg = f"Unsupported norder {norder}. "
            msg += "Supported values are between 0 and 5"
            raise ValueError(msg)
        
    @property
    def print_residuals(self) -> int:
        """Frequency of residuals print in output file

        0: never
        1: only after peak-constrained solution
        2: also after chosen solution
        3: always

        Returns
        -------
        int
            The print frequency mode for residuals
        """
        return self._print_res
    
    @property
    def print_fits(self) -> int:
        """Frequency of fits print in output file

        0: never
        1: only after peak-constrained solution
        2: also after chosen solution
        3: always

        Returns
        -------
        int
            The print frequency mode for fits
        """
        return self._print_fit
    
    @property
    def num_quad_grid_points(self) -> int:
        """Number of quadrature grid points

        Returns
        -------
        int
            The number of quadrature grid points
        """
        return self._num_quad_grid_pts
    
    @num_quad_grid_points.setter
    def num_quad_grid_points(self, num: int) -> None:
        """Set number of quadrature grid points

        Parameters
        ----------
        num : int
            The number of quadrature grid points (MUST be 2 or larger)

        Raises
        ------
        ValueError
            If num is less than 2
        """
        if num < 2:
            msg = "Number of quadrature grid points is too small, "
            msg += "num must be 2 or larger"
            raise ValueError(msg)
        else:
            self._num_quad_grid_pts = num

    @property
    def quad_grid_range(self) -> Tuple[float, float]:
        """Range of quadrature grid

        Returns
        -------
        Tuple[float, float]
            The range of the quadrature grid
        """
        return self._quad_grid_range
    
    @quad_grid_range.setter
    def quad_grid_range(self, grid_range: Tuple[float, float]) -> None:
        """Set range of quadrature grid

        NOTE: the values must be consistent with the units of the
        other quantities.

        Parameters
        ----------
        grid_range : Tuple[float, float]
            The range of the quadrature grid

        Raises
        ------
        ValueError
            If grid_range contains negative values
        """
        if min(grid_range) > 0:
            gmnmx1 = min(grid_range)
            gmnmx2 = max(grid_range)
            self._quad_grid_range = (gmnmx1, gmnmx2)
        else:
            msg = f"Unsupported grid range {grid_range}. "
            msg += "Range values must be positive"
            raise ValueError(msg)
        
    @property
    def modes(self) -> List[str]:
        """Supported analysis modes

        Returns
        -------
        List[str]
            The list of supported modes
        """
        return self._modes

    def set_data(
            self,
            k: float,
            tau: np.ndarray,
            isf: np.ndarray,
            err: Optional[np.ndarray] = None
    ) -> None:
        """Set data to be analyzed by CONTIN

        Parameters
        ----------
        k : float
            The wave vector magnitude
        tau : np.ndarray
            The delay times (in s)
        isf : np.ndarray
            The intermediate scattering function data
        err : Optional[np.ndarray], optional
            The uncertainty in the intermediate scattering function data,
            by default None. If None, an unweighted analysis will be performed

        Raises
        ------
        RuntimeError
            If the input arrays are not compatible
        """
        # sanity check
        # tau, isf, and err (if given) must have the same length
        # and they must be 1 dimensional
        len_tau = len(tau)
        dim_tau = len(tau.shape)
        len_isf = len(isf)
        dim_isf = len(isf.shape)

        if dim_tau == 1 and dim_isf == 1 and len_tau == len_isf:
            # delay times need to be sorted
            idx_sorted = np.argsort(tau)
            if err is not None:
                len_err = len(err)
                dim_err = len(err.shape)
                if dim_err == 1 and len_err == len_tau:
                    # sort error
                    self._err = err[idx_sorted]
                else:
                    msg = "err array must be 1D and have the same length as tau. "
                    msg += f"err array is {dim_err}-dimensional. "
                    msg += f"Lengths are err={len_err} and tau={len_tau}"
                    raise RuntimeError(msg)
            else:
                self._err = None
            # sort arrays
            self._tau_s = tau[idx_sorted]
            self._isf = isf[idx_sorted]
        else:
            msg = "Input arrays must be 1D and have the same length. "
            msg += "Dimensionalities of the arrays are "
            msg += f"tau={dim_tau}, isf={dim_isf}"
            raise RuntimeError(msg)
        
        self._k_cmm1 = k

    def _gen_param_str(
            self,
            name: str,
            value: Union[float, int, str],
            array_index: Optional[int] = None
    ) -> str:
        """Generate a parameter entry string for the CONTIN input file.

        Parameters
        ----------
        name : str
            The parameter name
        value : Union[float, int, str]
            The parameter value
        array_index : int, optional
            The parameter array index, by default None

        Returns
        -------
        str
            The CONTIN input file parameter string
        """
        special_parameters = {"IFORMT", "IFORMY", "IFORMW"}

        if name in special_parameters:
            return f" {name:6}\n {value}\n"
        else:
            # if index array is None, replace with empty string
            if array_index is None:
                array_index = ""
            if isinstance(value, int):
                return f" {name:6}{array_index:5}{value:>14}.\n"
            if isinstance(value, str):
                return f" {name:6}{array_index:5}{value:>15}\n"
            else:
                return f" {name:6}{array_index:5}{value:>15.6E}\n"

    def _gen_header_common(self) -> str:
        """Generate header string common for all analysis modes

        Sets by default the following CONTIN input parameters:
        - LAST      = TRUE  = only 1 analysis
        - NINTT     = 0     = no time intervals
        - NLINF     = 0
        - DOUSNQ    = TRUE  = enable constraint
        - NONNEG    = TRUE  = fix non-negative solution
        - RUSER(10) = 0     = leave isf unchanged

        The following parameters are also set:
        - the two extremes of the quadrature grid, GMNMX(1) and GMNMX(2)
        - the number of quadrature grid points, NG
        - the regularizor order, NORDER
        - frequency with which residuals are printed, IPLRES
        - frequency with which fits are printed, IPLFIT
        - use weights if uncertainty is provided, IWT

        Returns
        -------
        str
            The header string
        """
        head_str = ""

        # set LAST to 1
        head_str += self._gen_param_str(name="LAST", value=1)

        # set NG
        head_str += self._gen_param_str(name="NG",
                                        value=self.num_quad_grid_points)
        
        # set NINTT to 0
        head_str += self._gen_param_str(name="NINTT", value=0)

        # set NLINF to 0
        head_str += self._gen_param_str(name="NLINF", value=0)

        # set DOUSNQ to 1
        head_str += self._gen_param_str(name="DOUSNQ", value=1)

        # set NONNEG to 1
        head_str += self._gen_param_str(name="NONNEG", value=1)

        # set NORDER
        head_str += self._gen_param_str(name="NORDER", value=self.norder)

        # set IPLRES
        head_str += self._gen_param_str(name="IPLRES",
                                        value=self.print_residuals,
                                        array_index=2)
        
        # set IPLFIT
        head_str += self._gen_param_str(name="IPLFIT",
                                        value=self.print_fits,
                                        array_index=2)
        
        # set IWT
        iwt = 1 if self.err is None else 4
        head_str += self._gen_param_str(name="IWT", value=iwt)

        # set IFORMT, IFORMY, and IFORMW
        # NOTE: the format must mirror the one used in _gen_data_input
        head_str += self._gen_param_str(name="IFORMT",
                                        value=_DATA_FORMAT_FORTRAN)
        head_str += self._gen_param_str(name="IFORMY",
                                        value=_DATA_FORMAT_FORTRAN)
        if self.err is not None:
            head_str += self._gen_param_str(name="IFORMW",
                                            value=_DATA_FORMAT_FORTRAN)
            
        # set RUSER(10) to 0
        head_str += self._gen_param_str(name="RUSER",
                                        value=0.0,
                                        array_index=10)
        
        return head_str
    
    def gen_grid_range_time(self) -> Tuple[float, float]:
        """Estimate quadrature grid range for distribution of relaxation times

        Returns
        -------
        Tuple[float, float]
            The quadrature grid range
        """
        return self.tau_s[0], self.tau_s[-1]
    
    def _gen_header_time(self) -> str:
        """Write header for distribution of relaxation times

        If quad_grid_range is None, it uses gen_grid_range_time
        under the covers to estimate the quadrature grid range.

        Returns
        -------
        str
            The header string
        """
        head_str = ""

        # set GMNMX
        # if not given, set grid range for time
        if self.quad_grid_range is None:
            gmnmx1, gmnmx2 = self.gen_grid_range_time()
        else:
            gmnmx1, gmnmx2 = self.quad_grid_range
        head_str += self._gen_param_str(name="GMNMX",
                                        value=gmnmx1,
                                        array_index=1)
        head_str += self._gen_param_str(name="GMNMX",
                                        value=gmnmx2,
                                        array_index=2)
        
        # use generic distribution
        head_str += self._gen_param_str(name="IUSER",
                                        value=4,
                                        array_index=10)
        
        # set user values
        head_str += self._gen_param_str(name="RUSER",
                                        value=1.0,
                                        array_index=21)
        head_str += self._gen_param_str(name="RUSER",
                                        value=-1.0,
                                        array_index=22)
        head_str += self._gen_param_str(name="RUSER",
                                        value=0.0,
                                        array_index=23)
        head_str += self._gen_param_str(name="LUSER",
                                        value=-1,
                                        array_index=3)
        
        return head_str
    
    def gen_grid_range_diffusion(self) -> Tuple[float, float]:
        """Estimate quadrature grid range for distribution of diffusion
        coefficients

        Returns
        -------
        Tuple[float, float]
            The quadrature grid range
        """
        t_min, t_max = self.gen_grid_range_time()
        k2 = self.k_cmm1**2

        D_min = 1 / (t_max * k2)
        D_max = 1 / (t_min * k2)

        return D_min, D_max
    
    def _gen_header_diffusion(self) -> str:
        """Write header for distribution of diffusion coefficients

        If quad_grid_range is None, it uses gen_grid_range_diffusion
        under the covers to estimate the quadrature grid range.

        Returns
        -------
        str
            The header string
        """
        head_str = ""

        # set GMNMX
        # if not given, set grid range for diffusion
        if self.quad_grid_range is None:
            gmnmx1, gmnmx2 = self.gen_grid_range_diffusion()
        else:
            gmnmx1, gmnmx2 = self.quad_grid_range
        head_str += self._gen_param_str(name="GMNMX",
                                        value=gmnmx1,
                                        array_index=1)
        head_str += self._gen_param_str(name="GMNMX",
                                        value=gmnmx2,
                                        array_index=2)
        
        # use generic distribution
        # there is also a preset for diffusion coefficient distribution
        # but it computes the wave vector from the physical parameters
        # we use directly k instead
        head_str += self._gen_param_str(name="IUSER",
                                        value=4,
                                        array_index=10)
        
        # set user values
        head_str += self._gen_param_str(name="RUSER",
                                        value=self.k_cmm1**2,
                                        array_index=21)
        head_str += self._gen_param_str(name="RUSER",
                                        value=1.0,
                                        array_index=22)
        head_str += self._gen_param_str(name="RUSER",
                                        value=0.0,
                                        array_index=23)
        head_str += self._gen_param_str(name="LUSER",
                                        value=-1,
                                        array_index=3)

        return head_str
    
    def gen_grid_range_radius(self) -> Tuple[float, float]:
        """Estimate quadrature grid range for distribution of spheres
        satisfying the Stokes-Einstein relation

        Returns
        -------
        Tuple[float, float]
            The quadrature grid range
        """
        D_min, D_max = self.gen_grid_range_diffusion()
        kB = 1.38e-16
        T = self.temperature_K
        eta = 1e-2 * self.viscosity_cP

        R_min = kB * T / (6 * np.pi * eta * D_max)
        R_max = kB * T / (6 * np.pi * eta * D_min)

        return R_min, R_max
    
    def _gen_header_radius(self) -> str:
        """Write header for weight fraction distribution of spheres
        satisfying the Stokes-Einstein relation

        If quad_grid_range is None, it uses gen_grid_range_radius
        under the covers to estimate the quadrature grid range.

        Returns
        -------
        str
            The header string
        """
        head_str = ""

        # set GMNMX
        # if not given, set grid range for radius
        if self.quad_grid_range is None:
            gmnmx1, gmnmx2 = self.gen_grid_range_radius()
        else:
            gmnmx1, gmnmx2 = self.quad_grid_range
        head_str += self._gen_param_str(name="GMNMX",
                                        value=gmnmx1,
                                        array_index=1)
        head_str += self._gen_param_str(name="GMNMX",
                                        value=gmnmx2,
                                        array_index=2)
        
        # set weight fraction radius distribition of spheres
        # satisfying the Stokes-Einstein relation
        head_str += self._gen_param_str(name="RUSER",
                                        value=3.0,
                                        array_index=10)
        
        # set user values
        head_str += self._gen_param_str(name="RUSER",
                                        value=self.refractive_index,
                                        array_index=15)
        head_str += self._gen_param_str(name="RUSER",
                                        value=self.wavelength_nm,
                                        array_index=16)
        head_str += self._gen_param_str(name="RUSER",
                                        value=self.temperature_K,
                                        array_index=18)
        head_str += self._gen_param_str(name="RUSER",
                                        value=self.viscosity_cP,
                                        array_index=19)
        head_str += self._gen_param_str(name="RUSER",
                                        value=self.hollow_sphere_wall_thickness_cm,
                                        array_index=24)
        # we must provide the scattering angle in degrees
        # internally, it is computed as
        # q = 4e7 * pi * R15 / R16 * sin(R17 / 2)
        # therefore we need to convert our k to obtain k == q
        sin_thetad2 = self.k_cmm1 * self.wavelength_nm
        sin_thetad2 /= (4e7 * np.pi * self.refractive_index)
        theta_rad = 2 * np.arcsin(sin_thetad2)
        theta_deg = np.rad2deg(theta_rad)
        head_str += self._gen_param_str(name="RUSER",
                                        value=theta_deg,
                                        array_index=17)
        
        return head_str
    
    def _gen_data_input(self) -> str:
        """Write input data for analysis

        Returns
        -------
        str
            The data section string
        """
        data_str = ""

        # write length of data
        data_str += self._gen_param_str(name="NY",
                                        value="",
                                        array_index=len(self.tau_s))
        
        # concatenate data
        full_data = np.concatenate((self.tau_s, self.isf))
        if self.err is not None:
            full_data = np.concatenate((full_data, 1/self.err**2))
        
        # write data to string
        for data in full_data:
            data_str += f"{data:{_DATA_FORMAT}}\n"

        return data_str
    
    def write_input(self) -> None:
        """Wrap functions to write CONTIN input file for analysis

        Raises
        ------
        RuntimeError
            If data is not provided
        """
        if self.tau_s is None:
            raise RuntimeError("Data for analysis not provided")
        with open(self.file_in, "w") as fh:
            file_content = ""

            # write common header
            file_content += self._gen_header_common()

            # write mode specific header
            _gen_header_mode = {
                "time": self._gen_header_time,
                "diffusion": self._gen_header_diffusion,
                "radius": self._gen_header_radius
                }
            file_content += _gen_header_mode[self.mode]()

            # close header
            file_content += " END\n"

            # write data section
            file_content += self._gen_data_input()

            # write to file
            fh.write(file_content)

    def run_contin(self) -> None:
        """Run CONTIN executable

        Executes the command:
        contin < {file_in} > {file_out}
        """
        # generate command
        command = self.exec_path
        command += " < " + self.file_in
        command += " > " + self.file_out

        # execute command
        os.system(command)

    def _read_alpha_line(self, line: str) -> Tuple[float, ...]:
        """Read alpha line

        The line contains the following items:
        - ALPHA
        - ALPHA/S(1)
        - OBJ. FCTN.
        - VARIANCE
        - STD. DEV.
        - DEG. FREEDOM
        - PROB1 TO REJECT
        - PROB2 TO REJECT

        Parameters
        ----------
        line : str
            The line

        Returns
        -------
        Tuple[float, ...]
            The data
        """
        data = [float(d) for d in line.split()]

        return tuple(data)

    def read_output(self) -> Tuple[Dict[float, ContinResult], int]:
        """Retrieve results from CONTIN output file

        Returns
        -------
        Tuple[Dict[float, ContinResult], int]
            Dictionary of results (keys are alpha values) and optimum alpha
        """
        # initialize dictionary with results
        results = {}
        with open(self.file_out, "r") as fh:
            for line in fh:
                chk_in_sect_1 = line[:len(_INPUT_DATA_SECTION_START1)] == _INPUT_DATA_SECTION_START1
                chk_in_sect_2 = line[:len(_INPUT_DATA_SECTION_START2)] == _INPUT_DATA_SECTION_START2
                if chk_in_sect_1 or chk_in_sect_2:
                    if "SQRTW" in set(line.split()):
                        step = 3
                    else:
                        step = 2
                    input_data = []
                    while True:
                        data_line = next(fh)
                        if data_line[:len(_INPUT_DATA_SECTION_END)] == _INPUT_DATA_SECTION_END:
                            break
                        else:
                            data = data_line.split()
                            for d in data[1::step]:
                                input_data.append(_fortran2py_num_fmt(d))
                    input_data = np.array(input_data)
                elif line[:len(_ALPHA_SECTION_START)] == _ALPHA_SECTION_START:
                    no_skip = True
                    alpha_line = next(fh)
                    if alpha_line.split()[0] != "*":
                        data = self._read_alpha_line(alpha_line)
                        alpha = data[0]
                        alpha_over_s1 = data[1]
                        obj_fctn = data[2]
                        var = data[3]
                        std = data[4]
                        deg_frdm = data[5]
                        p1_rej = data[6]
                        p2_rej = data[7]
                    else:
                        no_skip = False
                elif line[:len(_CM_SECTION_START)] == _CM_SECTION_START and no_skip:
                    lambda_m = []
                    c_m = []
                    err_c_m = []
                    while True:
                        data_line = next(fh)
                        if data_line[:len(_CM_SECTION_END)] == _CM_SECTION_END:
                            break
                        else:
                            data = data_line.split()[:3]
                            data[-1] = re.sub(r"\.*X\.*", "", data[-1])
                            c, e, l = (_fortran2py_num_fmt(d) for d in data)
                            lambda_m.append(l)
                            c_m.append(c)
                            err_c_m.append(e)
                    lambda_m = np.array(lambda_m)
                    c_m = np.array(c_m)
                    err_c_m = np.array(err_c_m)
                elif line[:len(_FIT_SECTION_START)] == _FIT_SECTION_START and no_skip:
                    tau = []
                    fit = []
                    while True:
                        data_line = next(fh)
                        if data_line[:len(_FIT_SECTION_END)] == _FIT_SECTION_END:
                            break
                        else:
                            data = data_line.split()[:2]
                            data[-1] = re.sub(r"[OX*]", "", data[-1])
                            f, t = (_fortran2py_num_fmt(d) for d in data)
                            fit.append(f)
                            tau.append(t)
                    fit = np.array(fit)
                    tau = np.array(tau)
                    residuals = input_data - fit
                    results[alpha] = ContinResult(alpha,
                                                  alpha_over_s1,
                                                  obj_fctn,
                                                  var,
                                                  std,
                                                  deg_frdm,
                                                  p1_rej,
                                                  p2_rej,
                                                  lambda_m,
                                                  c_m,
                                                  err_c_m,
                                                  tau,
                                                  fit,
                                                  residuals)
                
        return results, alpha

    def analyze(self) -> Tuple[Dict[float, ContinResult], int]:
        """Wrap functions to analyze data with CONTIN

        Returns
        -------
        Tuple[Dict[float, ContinResult], int]
            Dictionary of results (keys are alpha values) and optimum alpha
        """
        # write input file
        self.write_input()

        # run CONTIN
        self.run_contin()
        
        # read output
        results, alpha_opt = self.read_output()

        # read output file and return results
        return results, alpha_opt
