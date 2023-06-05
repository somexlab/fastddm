# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Mike Chen 

from .azimuthalaverage import AzimuthalAverage
from fastddm.fit import simple_structure_function, fit

def residual_analysis(azi_large, azi_bidisperse):
    """
    Analysis the residual with the fitting of only the large particles.
    """
    if azi_large.k != azi_bidisperse.k:
        raise Exception(
            "q vectors for the large and bidisperese data must match.")
    if azi_large.tau != azi_bidisperse.tau:
        raise Exception(
            "time delays for the large and bidisperese data must match.")
    
    k = azi_large.k
    time_delay = azi_large.tau
    residual = []
    
    for idx, data_large in enumerate(azi_large.data):
        model = fit(simple_structure_function, azi_large.tau, data_large)
        parameters = model.best_values
        fitting = simple_structure_function.eval(dt = time_delay, 
                                                 A = parameters['A'],
                                                 B = parameters['B'],
                                                 tau = parameters['tau'])
        data_bidisperse = azi_bidisperse.data[idx]
        residual.append(fitting - data_bidisperse)
        
    
    
    return residual
