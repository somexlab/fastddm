# Copyright (c) 2023-2023 University of Vienna, Enrico Lattuada, Fabian Krautgasser, and Roberto Cerbino.
# Part of FastDDM, released under the GNU GPL-3.0 License.
# Authors: Mike Chen 

from .azimuthalaverage import AzimuthalAverage
from fastddm.fit import simple_structure_function, bidisperse_structure_function, fit

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


def plot_residual_heatmap(azi,
                          residual: Optional = None, 
                          xrange: Optional = None
                          yrange: Optional = None
                          vmin: Optional = None,
                          vmax: Optional = None,
                          log_scale: Optional = False):
    
    # qmin, qmax = min(azi.k), max(azi.k)
    # tmin, tmax = min(azi.tau), max(azi.tau)
    
    if residual is None:
        residual = relative_residual_analysis(azi)
    
    nk, ndata = np.shape(residual)
    
    if nk != len(azi.k):
        raise Exception("Number of bins not matched")
        
    if ndata != len(azi.tau):
        raise Exception("Time delay not matched")
        
    tt, kk = np.meshgrid(azi.tau,azi.k)
    
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
