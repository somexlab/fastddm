## Using the `dfmtoolbox.fit` module

The underlying framework in this module is [`lmfit`](https://lmfit.github.io//lmfit-py/). For convenience we provide a few helper tools for fitting the image structure function with a simple exponential function of the shape $D(q;\Delta t) = 2A(q)\left[1-\exp(\Delta t/\tau(q))\right] + 2B$.

We assume we have the 2 arrays for $\Delta t$ and $D(q;\Delta t)$ (for a fixed value of $q$). Then we can just fit the above shape for the image structure function:
```python
from dfmtoolbox.fit import simple_structure_function, fit

data = ... # the output of the azimuthal average
dt = np.logspace(-1, 2)  # here the considered lag times; dt.shape == data.shape

# the simple_structure_function has default values A=1.0, B=0.0, tau=1.0
result = fit(simple_structure_function, xdata=dt, ydata=data)

# and get the results with
best_fit = result.best_fit
best_values = result.best_values
```

(The `result` object is a `lmfit.model.ModelResult` instance.)

If there are starting parameters for the simple exponential model already obtained from elsewhere, they can be passed to the function as well. For this we first create a `lmfit.Parameters` object with the `<model>.make_params()` method, and directly pass the initial values:
```python
# get default parameters of the simple exponential model and adjust 2 of the initial starting values
parameters = simple_structure_function.make_params(A=42.0, B=6.9)

# the following procedure is the same, we just have to provide the params option
result = fit(simple_structure_function, xdata=dt, ydata=data, params=parameters)
best_fit = result.best_fit
best_values = ...
```

To just use basic estimations of initial values for $A, B, \tau$ one can also pass the `estimate_simple_parameters` option:
```python
data = ...
dt = ...

result = fit(simple_structure_function, xdata=dt, ydata=data, estimate_simple_parameters=True)
```
This will estimate some initial values, but it's only tested for the `simple_structure_function` model, and is not recommended to be used otherwise.

Of course, the simple exponential shape of the image structure function is not sufficient for many cases, so to fit a different model, it has to be properly set up. For example, using a stretched/compressed exponential instead of a simple exponential:
```python
import lmfit as lm
import numpy as np
from dfmtoolbox.fit import fit

def structure_function(dt: np.ndarray, A: float, B: float, tau: float, delta: float) -> np.ndarray:

    def stretched_exp(dt, tau, delta):
        return np.exp(-(dt/tau)**delta)

    return 2*A*(1-stretched_exp(dt, tau, delta)) + 2*B

# initialize model object
sf_model = lm.Model(structure_function)
# setting min/max values & initial parameters; min/max values are
# by default -/+ np.inf
sf_model.set_param_hint("A", min=0.0, max=np.inf, value=1.0)
sf_model.set_param_hint("B", min=0.0, value=0.0)
sf_model.set_param_hint("tau", min=0.0, value=1.0)
sf_model.set_param_hint("delta", min=0.0, max=4.0, value=1.0)

# fitting data to model
data = ...
dt = ...

result = fit(structure_function, xdata=dt, ydata=data)
```
Here we can directly set the initial values we want in the parameter hints. Not all parameter hints have to be set, but their default values (by `lmfit` may be problematic depending on the model used).
