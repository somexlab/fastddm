# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Mike Chen 

from .azimuthalaverage import AzimuthalAverage
from fastddm.fit import simple_structure_function, fit
from fastddm.fit_models import double_exponential_model
from fastddm.noise_est import estimate_camera_noise
import matplotlib.pyplot as plt

def relative_residual_analysis(data):
    residual = np.zeros(shape = np.shape(data.data))
    for idx, k in enumerate(data.k):
        model = fit(simple_structure_function, 
                    xdata = data.tau,
                    ydata = data.data[idx])
        residual[idx] = np.divide(model.residual, model.eval(
                                                            dt = data.tau,
                                                            A = model.best_values['A'],
                                                            B = model.best_values['B'],
                                                            tau = model.best_values['tau']))
    return residual


def plot_residual_heatmap(data,
                          residual: Optional = None, 
                          xrange: Optional = None
                          yrange: Optional = None
                          vmin: Optional = None,
                          vmax: Optional = None,
                          log_scale: Optional = False):
    
    # qmin, qmax = min(data.k), max(data.k)
    # tmin, tmax = min(data.tau), max(data.tau)
    
    if residual is None:
        residual = relative_residual_analysis(data)
    
    nk, ndata = np.shape(residual)
    
    if nk != len(data.k):
        raise Exception("Number of bins not matched")
        
    if ndata != len(data.tau):
        raise Exception("Time delay not matched")
        
    tt, kk = np.meshgrid(data.tau,data.k)
    
    residual = residual[:-1,:-1]
    if log_scale:
        residual = np.log(residual)
    if vmin != None or vmax != None:
        plt.pcolormesh(kk,tt, residual, vmin = vmin, vmax = vmax)
    else:
        plt.pcolormesh(kk,tt, residual)
    
    plt.colorbar()
    plt.xlabel('q (um^-1)')
    plt.ylabel(r'$\Delta t$ (s)')
    
    return
