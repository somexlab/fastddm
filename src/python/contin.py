# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

from typing import Union, Optional, Tuple, TextIO
import os
import numpy as np


def _generate_input_file(
        file: str,
        xdata: np.ndarray,
        ydata: np.ndarray,
        k: float,
        quad_grid_range: Tuple[float, float],
        errdata: Optional[np.ndarray] = None,
        mode: str = 'diffusion',
        **kwargs
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
        _write_header(fh, k, quad_grid_range, mode, **kwargs)
        # write data
        _write_data(fh, xdata, ydata, errdata)


def _generate_parameter_string(
        name: str,
        value: Union[float, int, str],
        array_index: Optional[int] = None
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
        num_quad_grid: int = 40,
        norder: int = 2,
        print_residuals: int = 3,
        print_fit: int = 3,
        use_err: bool = True,
        **kwargs
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
    par_str += _generate_parameter_string(name='GMNMX',
                                          value=min(quad_grid_range),
                                          array_index=1)
    par_str += _generate_parameter_string(name='GMNMX',
                                          value=max(quad_grid_range),
                                          array_index=2)

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
        par_str += _generate_parameter_string(name='IFORMW', value='(1E15.8)')

    # set RUSER(10) not to scale data
    par_str += _generate_parameter_string(name='RUSER', value=0, array_index=10)

    # write to file
    fh.write(par_str)


def _write_header_diffusion(
        fh: TextIO,
        k: float,
        **kwargs
        ) -> None:
    """Write header section for diffusion

    Parameters
    ----------
    fh : TextIO
        File handle
    k : float
        Magnitude of wave vector
    """
    # initialize parameter string
    par_str = ""

    # set generic distribution
    par_str += _generate_parameter_string(name='IUSER',
                                          value=4,
                                          array_index=10)

    # set user values
    par_str += _generate_parameter_string(name='RUSER',
                                          value=k**2,
                                          array_index=21)
    par_str += _generate_parameter_string(name='RUSER',
                                          value=1,
                                          array_index=22)
    par_str += _generate_parameter_string(name='RUSER',
                                          value=0,
                                          array_index=23)
    par_str += _generate_parameter_string(name='LUSER',
                                          value=0,
                                          array_index=3)

    # write to file
    fh.write(par_str)


def _write_header_stokes_einstein(
        fh: TextIO,
        k: float,
        refractive_index: float = 1.333,
        wavelength: float = 550,
        viscosity: float = 1,
        temperature: float = 300,
        wall_thickness: float = -1,
        **kwargs
        ) -> None:
    """Write header for weight fraction distribution of spheres
    satisfying the Stokes-Einstein relation

    Parameters
    ----------
    fh : TextIO
        File handle
    k : float
        Wave vector (in cm^-1)
    refractive_index : float, optional
        Medium refractive index, by default 1.333
    wavelength : float, optional
        Illumination wavelength (in nm), by default 550 nm
    viscosity : float, optional
        Medium viscosity (in cP = mPa s), by default 1 cP
    temperature : float, optional
        Absolute temperature (in K), by default 300 K
    wall_thickness : float, optional
        Wall thickness (in cm) for the hollow spheres Rayleigh-Debye form
        factor calculation. If negative, the solid spheres model is used.
        By default -1
    """
    # initialize parameter string
    par_str = ""

    # set weight fraction radius distribution of spheres
    # satisfying the Stokes-Einstein relation
    par_str += _generate_parameter_string(name='IUSER',
                                          value=3,
                                          array_index=10)

    # set user values
    # medium refractive index
    par_str += _generate_parameter_string(name='RUSER',
                                          value=refractive_index,
                                          array_index=15)
    # illumination wavelength (in nm)
    par_str += _generate_parameter_string(name='RUSER',
                                          value=wavelength,
                                          array_index=16)
    # scattering angle (in degrees)
    """
    the magnitude of the scattering vector is computed as
    q = 4 10^7 pi n sin(theta/2) / lambda
    so we need to compute the scattering angle (in degrees)
    """
    theta = np.arcsin(k * wavelength / (4e7 * np.pi * refractive_index))
    theta *= 180 / np.pi
    par_str += _generate_parameter_string(name='RUSER',
                                          value=theta,
                                          array_index=17)
    # absolute temperature (in K)
    par_str += _generate_parameter_string(name='RUSER',
                                          value=temperature,
                                          array_index=18)
    # viscosity (in cP)
    par_str += _generate_parameter_string(name='RUSER',
                                          value=viscosity,
                                          array_index=19)
    # wall thickness (in cm)
    par_str += _generate_parameter_string(name='RUSER',
                                          value=wall_thickness,
                                          array_index=24)

    # write to file
    fh.write(par_str)


def _write_header(
        fh: TextIO,
        k: float,
        quad_grid_range: Tuple[float, float],
        mode: str,
        **kwargs
        ) -> None:
    _write_header_common(fh=fh,
                         quad_grid_range=quad_grid_range,
                         **kwargs)

    _write_header_spec = {
        'diffusion': _write_header_diffusion,
        'stokes': _write_header_stokes_einstein
        }
    modes = _write_header_spec.keys()
    if mode in modes:
        _write_header_spec[mode](fh=fh, k=k, **kwargs)
    else:
        err_msg = f'Unknown mode {mode} selected. Supported modes are {modes}'
        raise RuntimeError(err_msg)

    # close header
    fh.write(" END\n")


def _write_data(
        fh: TextIO,
        xdata: np.ndarray,
        ydata: np.ndarray,
        errdata: Optional[np.ndarray] = None
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
