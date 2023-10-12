# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Author: Fabian Krautgasser
# Maintainer: Fabian Krautgasser

r"""This module contains a collection of helper fit functions.

The functions extensively use the `lmfit <https://lmfit.github.io/lmfit-py/>`_ package.

To fit the azimuthal average of the image structure function:

.. code-block:: python

    import fastddm as fddm

    # compute the image structure function and the azimuthal average (aa)
    ...

    # import fit model and do fit
    from fastddm.fit import fit_multik
    from fastddm.fit_models import simple_exponential_model as model

    # set the initial estimates for the reference k vector
    k_ref = 20  # this is the k index
    B_est, _ = fddm.noise_est.estimate_camera_noise(aa, mode='polyfit')
    model.set_param_hint('A', value=2.0 * aa.var[k_ref] - B_est[k_ref])
    model.set_param_hint('B', value=B_est[k_ref])
    model.set_param_hint('Gamma', value=...)

    result, model_results = fit_multik(aa,
                                       model,
                                       k_ref,
                                       use_err=True,
                                       return_model_results=True)

You can fix the parameters fit boundaries to some value, for instance:

.. code-block:: python

    A_max = 2.0 * aa.power_spec - B_est
    A_min = 2.0 * aa.var - B_est

    result, model_results = fit_multik(aa,
                                       model,
                                       k_ref,
                                       fixed_params_min={'A': A_min},
                                       fixed_params_max={'A': A_max})

Or you can fix the parameter directly, for instance:

.. code-block:: python

    result, model_results = fit_multik(aa,
                                       model,
                                       k_ref,
                                       fixed_params={'B': B_est})
"""

from typing import Any, Optional, Union, Dict, List, Tuple, Sequence

import lmfit as lm
import numpy as np
import pandas as pd

from .azimuthalaverage import AzimuthalAverage
from .intermediatescatteringfunction import IntermediateScatteringFunction


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
    return A * (1 - _simple_exp(dt, tau, 1.0)) + B  # type: ignore


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

    It is highly recommended to pass the ``weights`` argument for very noisy data. All keyword
    arguments in ``fitargs`` will directly be passed to the :py:meth:`lmfit.model.Model.fit`
    method (like weights). If ``params`` is presented and ``estimate_simple_parameters`` is set to True,
    first the simple parameters are estimated and the default model parameters updated, then
    the resulting parameters are updated with the presented parameters again.

    Parameters
    ----------
    model : lmfit.model.Model
        The model to be used for the fit.
    xdata : numpy.ndarray
        The data of the independent variable.
    ydata : numpy.ndarray
        The data we want to fit the model to.
    params : Optional[Union[lmfit.parameter.Parameters, lmfit.parameter.Parameter]], optional
        Either a single :py:class:`lmfit.parameter.Parameter` or
        :py:class:`lmfit.parameter.Parameters`, as the :py:class:`lmfit.model.Model` expects,
        by default None
    estimate_simple_parameters : bool
        Set to True if some simple estimates to get initial values for A, B and
        tau should be used. These estimates are only really applicable for the
        simple structure function setting, by default False
    verbose : bool, optional
        Pretty prints the parameters before fitting and the fit report
        afterwards, by default False

    Returns
    -------
    lmfit.model.ModelResult
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
    data: Union[AzimuthalAverage, IntermediateScatteringFunction],
    model: lm.Model,
    ref: int,
    ref_params: Optional[lm.Parameters] = None,
    return_model_results: Optional[bool] = False,
    use_err: Optional[bool] = False,
    fixed_params: Optional[Dict[str, Sequence[float]]] = None,
    fixed_params_min: Optional[Dict[str, Sequence[float]]] = None,
    fixed_params_max: Optional[Dict[str, Sequence[float]]] = None,
    fixed_params_expr: Optional[Dict[str, Sequence[str]]] = None,
    **fitargs: Any,
) -> Tuple[pd.DataFrame, Optional[List[lm.model.ModelResult]]]:
    r"""A wrapper for fitting a given model to given data for multiple :math:`k` vectors.

    The initial parameters estimated for ``ref`` index should be set in the model
    or passed to the function (via ``ref_params`` or as keyword arguments). All
    keyword arguments in ``fitargs`` will directly be passed to the
    :py:meth:`lmfit.model.Model.fit` method. See
    `here <https://lmfit.github.io/lmfit-py/model.html#lmfit.model.Model.fit>`_
    for more information. It is highly recommended to pass the ``weights``
    argument for very noisy data (must have the same size of ``data.tau``).
    Alternatively, one can pass True to the ``use_err`` flag and provide a ``data``
    input containing error values (in the ``data.err`` field).

    The function starts the fitting process from the ``ref`` index and proceeds
    towards lower and higher indices using the fit parameters obtained from the
    previous iteration.

    Note: If ``params`` is None, the values for all parameters relative to ``ref``
    are expected to be provided as keyword arguments. If ``params`` is given, and
    a keyword argument for a parameter value is also given, the keyword
    argument will be used. If neither ``params`` nor keyword arguments are given,
    the values set in the model will be used.

    Parameters
    ----------
    data : Union[AzimuthalAverage, IntermediateScatteringFunction]
        The azimuthal average or intermediate scattering function object to be fitted.
    model : lm.Model
        The model to be used for the fit. It must have one and one only
        independent variable (i.e., the time delay), the name is not important.
    ref : int
        The index of the reference `k` vector (where the initial fit parameters
        are estimated).
    ref_params : lmfit.parameter.Parameters, optional
        Parameters to use in fit in ``ref``. Default is None.
    return_model_results : bool, optional
        If True, the function also returns the complete list of
        :py:class:`lmfit.model.ModelResult`\s obtained for each `k` vector. Default is False.
    use_err : bool, optional
        If True, the error estimates in the ``AzimuthalAverage.err`` are used
        in place of ``weights``. If the ``AzimuthalAverage`` has no computed ``err``,
        the default ``weights`` are used.
    fixed_params : Dict[str, Sequence[float]], optional
        Dictionary of ``{parameter_name: values_array}`` pairs of parameter
        values to fix during fitting. The ``values_array`` length must be equal
        to ``len(data.k)``. Default is None.
    fixed_params_min : Dict[str, Sequence[float]], optional
        Dictionary of ``{parameter_name: values_array}`` pairs of parameter bounded min
        values to fix during fitting. The ``values_array`` length must be equal
        to ``len(data.k)``. Default is None.
    fixed_params_max : Dict[str, Sequence[float]], optional
        Dictionary of ``{parameter_name: values_array}`` pairs of parameter bounded max
        values to fix during fitting. The ``values_array`` length must be equal
        to ``len(data.k)``. Default is None.
    fixed_params_expr : Dict[str, Sequence[str]], optional
        Dictionary of ``{parameter_name: exprs_array}`` pairs of parameter expressions
        to fix during fitting. The ``values_array`` length must be equal to ``len(data.k)``.
        Default is None.

    Returns
    -------
    Tuple[pandas.DataFrame, Optional[List[lmfit.model.ModelResult]]]
        The fit parameters obtained and the success boolean value as a pandas
        DataFrame. If ``return_model_results`` is True, the result is a tuple
        whose second value is the complete list of :py:class:`lmfit.model.ModelResult`\s
        obtained.
    """
    from copy import deepcopy

    # initialize model parameters
    model_params = model.make_params()

    # create deep copies of passed parameters:
    ref_params = deepcopy(ref_params) if ref_params is not None else None
    fixed_params = deepcopy(fixed_params) if fixed_params is not None else None
    fixed_params_min = (
        deepcopy(fixed_params_min) if fixed_params_min is not None else None
    )
    fixed_params_max = (
        deepcopy(fixed_params_max) if fixed_params_max is not None else None
    )
    fixed_params_expr = (
        deepcopy(fixed_params_expr) if fixed_params_expr is not None else None
    )

    # we require the models to have one and one only independent variable
    indep_var = model.independent_vars[0]
    fitargs[indep_var] = data.tau  # mapping tau to independent variable name

    # initialize parameters
    # check ref_params
    if ref_params is not None:
        model_params.update(ref_params)
    # check **fitargs
    for p in model.param_names:
        if p in fitargs:
            # update parameter initial value
            model_params[p].value = fitargs[p]
            # delete the value so that it does not override
            # the one used during the iterations
            del fitargs[p]

    # fix parameters in ref
    if fixed_params is not None:
        for p, pval in fixed_params.items():
            model_params[p].vary = False
            model_params[p].value = pval[ref]
    if fixed_params_min is not None:
        for p, pval in fixed_params_min.items():
            model_params[p].min = pval[ref]
    if fixed_params_max is not None:
        for p, pval in fixed_params_max.items():
            model_params[p].max = pval[ref]
    if fixed_params_expr is not None:
        for p, pexpr in fixed_params_expr.items():
            model_params[p].expr = pexpr[ref]

    # initialize outputs
    results = {p: np.zeros(len(data.k)) for p in model.param_names}
    results.update({f'{p}_stderr': np.zeros(len(data.k)) for p in model.param_names})
    results["success"] = np.zeros(len(data.k), dtype=bool)
    results["k"] = data.k

    model_results = None
    if return_model_results:
        model_results = [None] * len(data.k)

    # perform fit in ref
    result = model.fit(data.data[ref], params=model_params, **fitargs)
    for p in model.param_names:
        results[p][ref] = result.params[p].value
        results[f'{p}_stderr'][ref] = result.params[p].stderr

    results["success"][ref] = result.success
    if model_results is not None:
        model_results[ref] = result

    def _fit(indexrange):
        # access nonlocal variables
        nonlocal data, ref, use_err, fixed_params, fixed_params_max, fixed_params_min, fitargs
        nonlocal model_params, model, results, model_results

        # update parameters
        for p in model.param_names:
            model_params[p].value = results[p][ref]

        # perform fit in indexrange
        for idx in indexrange:
            # fit
            if np.isnan(data.data[idx, 0]):
                for p in model.param_names:
                    results[p][idx] = np.nan
                    results[f'{p}_stderr'][idx] = np.nan
            else:
                # is use_err, set weights using error
                if use_err and data.err is not None:
                    fitargs["weights"] = 1.0 / data.err[idx]
                # if fixed_params_* is given, fix the parameters
                if fixed_params is not None:
                    for p, pval in fixed_params.items():
                        model_params[p].value = pval[idx]
                if fixed_params_min is not None:
                    for p, pval in fixed_params_min.items():
                        model_params[p].min = pval[idx]
                if fixed_params_max is not None:
                    for p, pval in fixed_params_max.items():
                        model_params[p].max = pval[idx]
                if fixed_params_expr is not None:
                    for p, pexpr in fixed_params_expr.items():
                        model_params[p].expr = pexpr[idx]
                # do fit
                result = model.fit(data.data[idx], params=model_params, **fitargs)
                # update results and model_params
                for p in model.param_names:
                    results[p][idx] = result.params[p].value
                    results[f'{p}_stderr'][idx] = result.params[p].stderr
                    model_params[p].value = result.params[p].value
                results["success"][idx] = result.success
                if model_results is not None:
                    model_results[idx] = result

    # perform fit towards small k vectors
    _fit(reversed(range(ref)))

    # perform fit towards large k vectors
    # also repeat fit at ref
    _fit(range(ref, len(data.k)))

    return pd.DataFrame(results), model_results
