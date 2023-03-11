# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Fabian Krautgasser
# Maintainer: Fabian Krautgasser

"""A collection of lmfit Models and helper/wrapper functions."""

from typing import Any, Optional, Union, Dict, List, Tuple

import lmfit as lm
import numpy as np

from .azimuthalaverage import AzimuthalAverage


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
    dt: np.ndarray, A: float, B: float, tau: float
) -> np.ndarray:
    """Basic image structure function shape with a simple exponential."""
    return 2 * A * (1 - _simple_exp(dt, tau, 1.0)) + 2 * B  # type: ignore


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
    If `params` is presented and `estimate_simple_parameters` is set to True, first the simple
    parameters are estimated and the default model parameters updated, then the resulting parameters
    are updated with the presented parameters again.

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
    # initialize model parameters
    model_params = model.make_params()

    # we will assume the models only have one indep. variable
    indep_var = model.independent_vars[0]
    fitargs[indep_var] = xdata  # mapping xdata to independent variable name

    # estimating simple parameters and updating model parameters
    if estimate_simple_parameters:
        simple_pars = _simple_structure_function_parameter_helper(xdata, ydata)
        if verbose:
            print(":: updating given parameters with simple estimates .. ")
        model_params.update(simple_pars)

    # updating model parameters with provided parameters
    if params is not None:
        model_params.update(params)
        if verbose:
            print(":: Model parameters:")
            model_params.pretty_print()

    # fit
    result = model.fit(ydata, params=model_params, **fitargs)

    if verbose:
        print(":: Fit report:")
        print(result.fit_report())

    return result


def fit_multik(
    data: AzimuthalAverage,
    model: lm.Model,
    ref: int,
    weights: Optional[np.ndarray] = None,
    return_model_results : Optional[bool] = False,
    **fitargs: Any,
) -> Tuple[Dict[str, np.ndarray], Optional[List[lm.model.ModelResult]]]:
    """A wrapper for fitting a given model to given data for multiple k vectors.

    The initial parameters estimated for `ref` index should be set or passed to the function
    (via `params` or as keyword arguments). It is highly recommended to pass the `weights`
    argument for very noisy data. All keyword arguments in `fitargs` will directly be passed
    to the lm.Model.fit method. See
    https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.fit
    for more information.

    The function starts the fitting process from the `ref` index and proceeds towards
    smaller and larger indices using the fit parameters obtained from the previous
    iteration.

    Parameters
    ----------
    data : AzimuthalAverage
        The azimuthal average object to be fitted.
    model : lm.Model
        The model to be used for the fit.
    ref : int
        The index of the reference k vector (where the initial fit parameters are estimated).
    weights : np.ndarray, optional
        Weights to use for the calculation of the fit residual [i.e., weights*(data-fit)].
        Default is None; must have the same size as data.tau.
    return_model_results : bool, optional
        If True, the function also returns the complete list of ModelResults obtained for each
        k vector. Default is False.

    Returns
    -------
    Tuple[Dict[str, np.ndarray], Optional[List[lm.model.ModelResult]]]
        The fit parameters obtained and the success boolean value as a dictionary of
        string keys and numpy arrays values. If `return_model_results` is True,
        the result is a tuple whose second value is the complete list of
        ModelResults obtained. 
    """

    # initialize outputs
    results = {p : np.zeros(len(data.k)) for p in model.param_names}
    results['success'] = np.zeros(len(data.k), dtype=False)

    model_results = None
    if return_model_results:
        model_results = [None] * len(data.k)

    # perform fit in ref
    result = model.fit(data.data[ref], x=data.tau, weights=weights, **fitargs)
    for p in model.param_names:
        results[p][ref] = result.params[p].value
    results['success'][ref] = result.success
    if return_model_results:
        model_results[ref] = result

    # perform fit towards small k vectors
    idx = ref
    # update parameters
    model_params = model.make_params()
    for p in model.param_names:
        model_params[p].value = results[p][ref]
    while idx > 0:
        # update index
        idx -= 1
        # fit
        if np.isnan(data.var[idx]):
            results[p][idx] = np.nan
        else:
            result = model.fit(data.data[idx], x=data.tau, params=model_params, weights=weights, **fitargs)
            # update results and model_params
            for p in model.param_names:
                model_params[p].value = results[p][idx] = result.params[p].value
            results['success'][idx] = result.success
            if return_model_results:
                model_results[idx] = result

    # perform fit towards large k vectors
    idx = ref + 1
    # update parameters
    for p in model.param_names:
        model_params[p].value = results[p][ref]
    while idx < len(data.k):
        # fit 
        if np.isnan(data.var[idx]):
            results[p][idx] = np.nan
        else:
            result = model.fit(data.data[idx], x=data.tau, params=model_params, weights=weights, **fitargs)
            # update results and model_params
            for p in model.param_names:
                model_params[p].value = results[p][idx] = result.params[p].value
            results['success'][idx] = result.success
            if return_model_results:
                model_results[idx] = result
        # update index
        idx += 1

    if return_model_results:
        return results, model_results
    else:
        return results
