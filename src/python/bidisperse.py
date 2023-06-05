# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Mike Chen 

from .azimuthalaverage import AzimuthalAverage
from fastddm.fit import simple_structure_function, fit

def relative_residual_analysis(azi):
    residual = np.zeros(shape = np.shape(azi.data))
    for idx, k in enumerate(azi.k):
        model = fit(simple_structure_function, 
                    xdata = azi.tau,
                    ydata = azi.data[idx])
        residual[idx] = np.divide(model.residual, model.eval(
                                                            dt = azi.tau,
                                                            A = model.best_values['A'],
                                                            B = model.best_values['B'],
                                                            tau = model.best_values['tau']))
    return residual
