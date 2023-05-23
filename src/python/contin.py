# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

from typing import Union, Optional, List, Tuple, TextIO
import os
import numpy as np

"""Parameters summary (see contin-manual1.pdf)
LAST:       if TRUE, CONTIN stops after analysis (1 = TRUE, 0 = FALSE)
GMNMX, 1:   first grid point in the quadrature
GMNMX, 2:   last point in the quadrature
IWT:        1 = unweighted analysis
            2 = error proportional to sqrt(y(tk)), i.e., Poisson statistics
            3 = error proportional to |y(tk)|, i.e., constant relative error
            4 = input wk in Card set 7
            5 = computing wk in USERWT
NERFIT:     number of residuals used to compute ERRFIT
NINTT:      number of intervals for tk.     (THIS SHOULD ALWAYS BE <=0)
NLINF:      NL in Eq. 3.1-2
IFORMY:     Fortran FORMAT specification enclosed in parentheses for yk
            in Card Set 6
IFORMT:     Fortran FORMAT specification enclosed in parentheses for tk
            in Card Set 5b
IFORMW:     Fortran FORMAT specification enclosed in parentheses for wk
            in Card Set 7
NG:         Number of quadrature grid points, if Eq.3.1-2 is solved by
            quadrature. Nx if Eq.3.1-1 is solved directly
DOUSNQ:     if TRUE, USERNQ is to be called to specify inequality constraints
            (1 = TRUE, 0 = FALSE)
USERNQ:     sets the inequality constraints in Eq.3.1-4
NONNEG:     if TRUE, constrains s(gm) in Eq.3.1-3 to be non-negative
            (1 = TRUE, 0 = FALSE)
NORDER:     choice of the regularizors
            < 0: for calling USERRG to set a special user-defined regularizor
            0-5: for setting the regularizor (sum in Eq.3.2.2-1) to be the sums
            of the squares of the nth differences of the Ng-n sets (similar to
            order of derivative). 0 is sum xj^2, 2 is sum of second derivative
            squared (smooth solution)
IPLRES(2):  controls when the weighted residuals will be plotted.
            0: never
            1: only after peak-constrained solution
            2: also after the CHOSEN SOLUTION
            3: after every solution
IPLFIT(2):  same as IPLRES, except that it controls when the plots of the fit to
            the data will be made.
RUSER(1):   s(lambda_1)
RUSER(2):   s(lambda_Ng)
RUSER(3):   noise level
RUSER(6):   integral of s(lambda)
RUSER(10):  0: yk are not changed
            < 0: yk are replaced by yk^1/2          (THIS SHOULD NEVER BE USED WITH DDM)
            > 0: yk are replaced by (yk/R10-1)^1/2  (THIS SHOULD NEVER BE USED WITH DDM)
RUSER(15):  medium refractive index.
RUSER(16):  wavelength illumination source (in nm).
            If R16=0, R20 is not computed and R21 is set to 0.
RUSER(17):  scattering angle (in degrees).
RUSER(18):  absolute temperature.
RUSER(19):  viscosity (in cP).
RUSER(20):  scattering vector (in cm^-1). Computed from R15, R16, R17.
RUSER(24):  wall thickness of hollo spheres (in cm).
IUSER(10):  1: s(lambda) is weight fraction molecular weight distribution.
               R23=1, R22=R18*R20^2, so that D=R18*lambda^R22
            2: s(lambda) is diffusion coefficient distribution.
               R23=0, R22=1, R21=R20^2
            3: s(lambda) is weight fraction radius distribution (in cm)
               of spheres satisfying the Stokes-Einstein relation.
               R23=3, R22=-1, R21=kB*R18*R20^2/(0.06*pi*R19)
            4: generic case where R21, R22, and R23 are set by the user
LUSER(3):   0: do not use form factors (i.e., f_m = 1)
            1: use Rayleigh Debye form factors for hollow spheres with R24 wall
               thickness (in cm). If R24<=0, the form factors for solid spheres
               are computed. An I18>0 causes the squared form factor to be
               averaged over 2*I18+1 equally spaced points on the interval
               centered at the grid point and extending halfway to its nearest
               neighbors (if form factor rapidly oscillates). Default I18=50 is
               recommended.

Kernels are of the form
F(lambda_m, tk) = f_m^2 lambda_m^R23 exp(-R21 tk lambda_m^R22)
"""


def _generate_input_file(
        file: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        errdata: Optional[np.ndarray]=None
        ) -> None:
    """Generate CONTIN input file with parameters and data.

    Parameters
    ----------
    file : str
        File name
    xdata : np.ndarray
        x data
    ydata : np.ndarray
        y data
    errdata : np.ndarray, optional
        Errors to be used for weighted analysis, by default None
    """
    with open(file, 'w') as fh:
        # write header
        # write data
        _write_data(fh, xdata, ydata, errdata)


def _generate_parameter_string(
        name: str,
        value: Union[float, int, str],
        array_index: Optional[int]=None
        ) -> str:
    """Generate a parameter string for the CONTIN input file.

    Parameters
    ----------
    name : str
        Parameter name
    value : Union[float, int, str]
        Parameter value
    array_index : int, optional
        Parameter array index, by default None

    Returns
    -------
    str
        CONTIN input file parameter string
    """
    special_parameters = ('IFORMT', 'IFORMY', 'IFORMW')

    if name in special_parameters:
        return f" {name:6}\n {value}\n"
    else:
        if array_index is None:
            array_index = ''
        if isinstance(value, float):
            return f" {name:6}{array_index:5}{value:>15.6E}\n"
        if isinstance(value, int):
            return f" {name:6}{array_index:5}{value:>15}.\n"


def _write_header_common(
        fh: TextIO,
        quad_grid_range: Tuple[float, float],
        num_quad_grid: int=40,
        norder: int=2,
        print_residuals: int=3,
        print_fit: int=3,
        use_err: bool=True
        ) -> None:
    """Write common header for all modes

    Sets by default the following CONTIN input parameters:
    - LAST = TRUE
    - NINTT = 0
    - NLINF = 1
    - DOUSNQ = TRUE
    - NONNEG = TRUE

    Other parameters can be set as follows:
    - quad_grid_range: sets the two extremes of the quadrature grid, GMNMX(1) and GMNMX(2)
    - num_quad_grid: sets the number of quadrature grid ponts, NG
    - norder: sets the regularizor order, NORDER
    - print_residuals: sets how often residuals are printed, IPLRES
    - print_fit: sets how often fits are printed, IPLFIT

    Parameters
    ----------
    fh : TextIO
        File handle
    quad_grid_range : Tuple[float, float]
        Range of the quadrature grid
    num_quad_grid : int, optional
        Number of quadrature grid points (minimum 2), by default 40
    norder : int, optional
        Regularizor order (analogous to derivative order). Must be between 0 and 5, by default 2
    print_residuals : int, optional
        How often residuals are printed in output file, by default 3
        0 = never
        1 = only after peak constrained solution
        2 = also after the chosen solution
        3 = always
    print_fit : int, optional
        How often residuals are printed in output file. Same options as print_residuals.
        Default is 3
    use_err : bool, optional
        Use errors for weighted analysis, by default True

    Raises
    ------
    ValueError
        If any of the input variables is out of range.
    """
    # initialize parameter string
    par_str = ""

    # set LAST to 1
    # just one analysis
    par_str += _generate_parameter_string(name='LAST', value=1)

    # set NG to num_quad_grid
    if num_quad_grid < 2:
        err_msg = f"Number of quadrature grid points {num_quad_grid} is too small.\n"
        err_msg += "Increase the number of points to be 2 or larger."
        raise ValueError(err_msg)
    par_str += _generate_parameter_string(name='NG', value=num_quad_grid)

    # set GMNMX 1 and 2
    if min(quad_grid_range) < 0:
        raise ValueError("Negative quadrature grid points are not accepted.")
    par_str += _generate_parameter_string(name='GMNMX', value=min(quad_grid_range), array_index=1)
    par_str += _generate_parameter_string(name='GMNMX', value=max(quad_grid_range), array_index=2)

    # set NINTT to 0
    # no time intervals, all times are listed
    par_str += _generate_parameter_string(name='NINTT', value=0)

    # set NLINF to 1
    par_str += _generate_parameter_string(name='NLINF', value=1)

    # set DOUSNQ to 1 (= TRUE)
    # enable constraints
    par_str += _generate_parameter_string(name='DOUSNQ', value=1)

    # set NONNEG to 1 (= TRUE)
    # fix non-negative solution
    par_str += _generate_parameter_string(name='NONNEG', value=1)

    # set NORDER to 2 (smooth solution)
    # 2 corresponds to sum of squared second derivatives
    if norder < 0 or norder > 5:
        err_msg = f"Order {norder} not supported.\n"
        err_msg += "Value must be between 0 and 5."
        raise ValueError(err_msg)
    par_str += _generate_parameter_string(name='NORDER', value=2)

    # set IWT
    if use_err:
        par_str += _generate_parameter_string(name='IWT', value=4)
    else:
        par_str += _generate_parameter_string(name='IWT', value=1)

    # set IPLRES and IPLFIT to 3
    if print_residuals < 0 or print_residuals > 3:
        err_msg = f"Residual print option {print_residuals} not supported.\n"
        err_msg += "Value must be between 0 and 3."
        raise ValueError(err_msg)
    par_str += _generate_parameter_string(name='IPLRES', value=print_residuals)
    if print_fit < 0 or print_fit > 3:
        err_msg = f"Fit print option {print_fit} not supported.\n"
        err_msg += "Value must be between 0 and 3."
        raise ValueError(err_msg)
    par_str += _generate_parameter_string(name='IPLFIT', value=print_fit)

    # set IFORMT, IFORMY (and IFORMW is use_err is True)
    # NOTE!! they must mirror the ones in _write_data()
    par_str += _generate_parameter_string(name='IFORMT', value='(1E15.8)')
    par_str += _generate_parameter_string(name='IFORMY', value='(1E15.8)')
    if use_err:
        par_str += _generate_input_file(name='IFORMW', value='(1E15.8)')

    # write to file
    fh.write(par_str)


def _write_header_diffusion(
        fh: TextIO,
        k: float
        ) -> None:
    pass    


def _write_header(
        fh: TextIO,
        quad_grid_range: Tuple[float, float],
        mode: str='diffusion',
        **kwargs
        ) -> None:
    
    # close header
    fh.write(" END\n")


def _write_data(
        fh: TextIO,
        xdata: np.ndarray,
        ydata: np.ndarray,
        errdata: Optional[np.ndarray]=None
        ) -> None:
    """Write CONTIN input data to file.

    Parameters
    ----------
    file : TextIO
        File handle
    xdata : np.ndarray
        x data
    ydata : np.ndarray
        y data
    errdata : np.ndarray, optional
        err data
    """
    # write length of data
    fh.write(_generate_parameter_string(name='NY', value=len(xdata)))

    # concatenate data
    full_data = np.concatenate((xdata, ydata))
    if errdata is not None:
        full_data = np.concatenate((full_data, 1/errdata))

    # write data to file
    for d in full_data:
        fh.write(f"{d:>15.8E}\n")


def _run_contin(
        input_file: str,
        output_file: str
        ) -> None:
    """Run CONTIN executable.

    Executes the command:
    contin < input_file > output_file

    Parameters
    ----------
    input_file : str
        Input file to CONTIN executable
    output_file : str
        File to which CONTIN output is redirected
    """
    this_file_path = os.path.abspath(os.path.dirname(__file__))
    contin_exec_path = os.path.join(this_file_path, "contin")
    exec_file = contin_exec_path
    
    full_comm = exec_file + " < " + input_file + " > " + output_file

    os.system(full_comm)


