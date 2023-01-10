"""A collection of lmfit Models and helper/wrapper functions."""

from typing import Any, Optional, Union

import lmfit as lm
import numpy as np


def _simple_exp(
    dt: Union[np.ndarray, float, int], tau: float, amplitude: float
) -> Union[np.ndarray, float]:
    """A simple exponential of the shape
    simple_exp(dt, tau, amplitude) = amplitude * exp(-dt/tau)
    """

    return amplitude * np.exp(-dt / tau)


simple_exp_model = lm.Model(_simple_exp)
simple_exp_model.set_param_hint("tau", min=0.0, max=np.inf)
simple_exp_model.set_param_hint(
    "amplitude", min=0.0, max=2.0
)  # max value of 1 possibly causes artifacts


def _simple_image_structure_function(
    dt: np.ndarray,
    A: float,
    B: float,
    tau: float
) -> np.ndarray:
    """Basic image structure function shape with a simple exponential."""
    return 2 * A * (1 - _simple_exp(dt, tau, 1)) + 2 * B


simple_structure_function = lm.Model(_simple_image_structure_function)
simple_structure_function.set_param_hint("A", min=0.0, max=np.inf, value=1.0)
simple_structure_function.set_param_hint("B", min=0.0, max=np.inf, value=0.0)
simple_structure_function.set_param_hint("tau", min=0.0, max=np.inf, value=1.0)


def _simple_structure_function_parameter_helper(
    xdata: np.ndarray,
    ydata: np.ndarray,
) -> lm.Parameters:

    # first estimate B from the constant offset of a 2nd-degree polynomial fit of the first values
    n = 5
    poly = np.polynomial.Polynomial.fit(xdata[:n], ydata[:n], deg=2)
    B = max(poly.coef[0], 0.0)  # first coefficient is constant offset; ensure B >= 0.0

    # estimate A
    A = max(ydata[-1] - B, 0.0)  # ensure A >= 0.0

    # estimate tau
    if A > 0.0:
        tau = xdata[np.argmin(np.abs((ydata - B) / A - 0.5))]
    else:
        tau = 1.0

    params = lm.Parameters()
    for name, val in {"A": A, "B": B, "tau": tau}.items():
        params.add(name, min=0.0, max=np.inf, value=val)

    return params


def fit(
    model: lm.Model,
    xdata: np.ndarray,
    ydata: np.ndarray,
    params: Optional[Union[lm.Parameters, lm.Parameter]] = None,
    estimate_simple_parameters: bool = False,
    verbose: bool = False,
    **fitargs: Any,
) -> lm.model.ModelResult:
    """A wrapper for fitting a given model to given data.

    It is highly recommended to pass the `weights` argument for very noisy data. All keyword
    arguments in `fitargs` will directly be passed to the lm.Model.fit method (like weights).

    Parameters
    ----------
    model : lm.Model
        The model to be used for the fit.
    xdata : np.ndarray
        The data of the independent variable.
    ydata : np.ndarray
        The data we want to fit the model to.
    params : Optional[Union[lm.Parameters, lm.Parameter]], optional
        Either a single lm.Parameter or lm.Parameters, as the Model expects, by
        default None
    estimate_simple_parameters : bool
        Set to true if some simple estimates to get initial values for A, B and
        tau should be used. These estimates are only really applicable for the
        simple structure function setting, by default False
    verbose : bool, optional
        Pretty prints the parameters before fitting and the fit report
        afterwards, by default False

    Returns
    -------
    lm.model.ModelResult
        The results of the fit.
    """
    params = model.make_params() if params is None else params
    if verbose:
        print(":: Model parameters:")
        params.pretty_print()

    # we will assume the models only have one indep. variable
    indep_var = model.independent_vars[0]
    fitargs[indep_var] = xdata  # mapping xdata to independent variable name

    if estimate_simple_parameters:
        simple_pars = _simple_structure_function_parameter_helper(xdata, ydata)
        if verbose:
            print(":: updating given parameters with simple estimates .. ")
        params.update(simple_pars)

    # fit
    result = model.fit(ydata, params=params, **fitargs)

    if verbose:
        print(":: Fit report:")
        print(result.fit_report())

    return result
