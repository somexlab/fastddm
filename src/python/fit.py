"""A collection of lmfit Models and helper/wrapper functions."""

import lmfit as lm
import numpy as np
from typing import Union, Any, Optional


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


def fit(
    model: lm.Model,
    xdata: np.ndarray,
    ydata: np.ndarray,
    params: Optional[Union[lm.Parameters, lm.Parameter]] = None,
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
        Either a single lm.Parameter or lm.Parameters, as the Model expects, by default None
    verbose : bool, optional
        Pretty prints the parameters before fitting and the fit report afterwards, by default False

    Returns
    -------
    lm.model.ModelResult
        The results of the fit.
    """
    if verbose:
        print(":: Model parameters:")
        p = model.make_params() if params is None else params
        p.pretty_print()

    # we will assume the models only have one indep. variable
    indep_var = model.independent_vars[0]
    fitargs[indep_var] = xdata  # mapping xdata to independent variable name

    # fit
    result = model.fit(ydata, params=params, **fitargs)

    if verbose:
        print(":: Fit report:")
        print(result.fit_report())

    return result
