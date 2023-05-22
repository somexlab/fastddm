# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Enrico Lattuada
# Maintainers: Enrico Lattuada

from typing import Union, Optional, List, TextIO
import os
import numpy as np

from fastddm.imagestructurefunction import ImageStructureFunction
from fastddm.azimuthalaverage import AzimuthalAverage

this_file_path = os.path.abspath(os.path.dirname(__file__))
contin_exec_path = os.path.join(this_file_path, "contin")

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
NINTT:      number of intervals for tk
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
            < 0: yk are replaced by yk^1/2
            > 0: yk are replaced by (yk/R10-1)^1/2
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
        xdata: Union[np.ndarray, List],
        ydata: Union[np.ndarray, List]
        ) -> None:
    """Generate contin input file with parameters and data.

    Parameters
    ----------
    file : str
        File name
    xdata : Union[np.ndarray, List]
        x data
    ydata : Union[np.ndarray, List]
        y data
    """
    with open(file, 'w') as fh:
        # write header
        # write data
        _write_data(fh, xdata, ydata)



    


def _write_header(
        fh: TextIO,
        mode: str='diffusion',
        **kwargs
        ) -> None:
    pass


def _write_data(
        fh: TextIO,
        xdata: Union[np.ndarray, List],
        ydata: Union[np.ndarray, List]
        ) -> None:
    """Write contin input data to file.

    Parameters
    ----------
    file : TextIO
        File handle
    xdata : Union[np.ndarray, List]
        x data
    ydata : Union[np.ndarray, List]
        y data
    """
    full_data = np.concatenate((xdata, ydata))

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
        Input file to contin executable
    output_file : str
        File to which contin output is redirected
    """
    exec_file = contin_exec_path
    full_comm = exec_file + " < " + input_file + " > " + output_file

    os.system(full_comm)


